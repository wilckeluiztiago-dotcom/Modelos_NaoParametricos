# -*- coding: utf-8 -*-
"""
Módulo 03 — Sub-bandas Quânticas e Confinamento Transversal
===========================================================

Resolve o problema de autovalores da equação de Schrödinger 2D no plano
transversal (y,z) do nanofio retangular, obtendo as energias das sub-bandas
E_{n,m} e as funções de onda φ_{n,m}(y,z).

Autor: Luiz Tiago Wilcke
Referência: Cap. 36 + física de semicondutores do livro (2026).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh

from constantes_fisicas import CONSTANTES, converter_joule_para_ev
from parametros_dispositivo import ParametrosDispositivo, criar_dispositivo_padrao


@dataclass
class SubBanda:
    """Representa uma única sub-banda quantizada."""
    indices: Tuple[int, int]          # (n, m)
    energia_joule: float
    energia_ev: float
    degenerescencia: int = 1
    funcao_onda: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"SubBanda(n={self.indices[0]}, m={self.indices[1]}, E={self.energia_ev:.4f} eV)"


class CalculadorSubBandas:
    """
    Calcula as energias e funções de onda das sub-bandas de um nanofio
    retangular com paredes infinitas (aproximação de poço de potencial infinito).

    Para paredes infinitas a solução analítica é:

        E_{n,m} = (ħ² π² / 2 m*) (n² / W² + m² / H²)

        φ_{n,m}(y,z) = sqrt(4/(W H)) sin(n π y / W) sin(m π z / H)
    """

    def __init__(self, dispositivo: Optional[ParametrosDispositivo] = None):
        self.dispositivo = dispositivo or criar_dispositivo_padrao()
        self.massa_efetiva = self.dispositivo.material.massa_efetiva_kg
        self.largura = self.dispositivo.geometria.largura_m
        self.altura = self.dispositivo.geometria.altura_m
        self.hbar = CONSTANTES.hbar
        self._cache_energias: Dict[Tuple[int, int], float] = {}

    def energia_analitica(self, n: int, m: int) -> float:
        """
        Energia analítica da sub-banda (n,m) em joules.
        """
        if n < 1 or m < 1:
            raise ValueError("Índices de sub-banda devem ser ≥ 1.")
        chave = (n, m)
        if chave in self._cache_energias:
            return self._cache_energias[chave]

        termo_y = (n * math.pi / self.largura) ** 2
        termo_z = (m * math.pi / self.altura) ** 2
        energia = (self.hbar ** 2 / (2.0 * self.massa_efetiva)) * (termo_y + termo_z)
        self._cache_energias[chave] = energia
        return energia

    def energia_analitica_ev(self, n: int, m: int) -> float:
        """Energia em elétron-volts."""
        return converter_joule_para_ev(self.energia_analitica(n, m))

    def funcao_onda_analitica(
        self,
        n: int,
        m: int,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Função de onda normalizada φ_{n,m}(y,z) no domínio [0,W] × [0,H].
        """
        norm = math.sqrt(4.0 / (self.largura * self.altura))
        fy = np.sin(n * math.pi * y / self.largura)
        fz = np.sin(m * math.pi * z / self.altura)
        return norm * fy * fz

    def calcular_primeiras_subbandas(
        self,
        numero_n: int = 5,
        numero_m: int = 5,
    ) -> List[SubBanda]:
        """
        Calcula as primeiras numero_n × numero_m sub-bandas ordenadas por energia.
        """
        lista: List[SubBanda] = []
        for n in range(1, numero_n + 1):
            for m in range(1, numero_m + 1):
                e_j = self.energia_analitica(n, m)
                e_ev = converter_joule_para_ev(e_j)
                lista.append(SubBanda(indices=(n, m), energia_joule=e_j, energia_ev=e_ev))
        lista.sort(key=lambda s: s.energia_joule)
        return lista

    def energia_fundamental(self) -> SubBanda:
        """Retorna a sub-banda fundamental E_{1,1}."""
        e_j = self.energia_analitica(1, 1)
        return SubBanda(
            indices=(1, 1),
            energia_joule=e_j,
            energia_ev=converter_joule_para_ev(e_j),
            degenerescencia=1,
        )

    def resolver_numericamente_2d(
        self,
        pontos_y: int = 80,
        pontos_z: int = 80,
        numero_autovalores: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve numericamente o problema de autovalores 2D usando diferenças
        finitas e matriz esparsa (validação da solução analítica).

        Retorna
        -------
        energias_ev : ndarray
            Autovalores em eV.
        funcoes_onda : ndarray
            Autovetores (funções de onda) reformatados.
        """
        dy = self.largura / (pontos_y + 1)
        dz = self.altura / (pontos_z + 1)

        # Operador de Laplace 1D com condições de Dirichlet
        diag_prin = -2.0 * np.ones(pontos_y)
        diag_off = np.ones(pontos_y - 1)
        lap_y = diags([diag_off, diag_prin, diag_off], [-1, 0, 1], format="csr") / dy**2

        diag_prin_z = -2.0 * np.ones(pontos_z)
        diag_off_z = np.ones(pontos_z - 1)
        lap_z = diags([diag_off_z, diag_prin_z, diag_off_z], [-1, 0, 1], format="csr") / dz**2

        # Laplaciano 2D = I ⊗ L_y + L_z ⊗ I
        iy = eye(pontos_y, format="csr")
        iz = eye(pontos_z, format="csr")
        laplaciano_2d = kron(iz, lap_y) + kron(lap_z, iy)

        # Hamiltoniano H = - (ħ² / 2m) ∇²
        fator = -(self.hbar ** 2) / (2.0 * self.massa_efetiva)
        hamiltoniano = fator * laplaciano_2d

        # Autovalores e autovetores (menores energias)
        autovalores, autovetores = eigsh(
            hamiltoniano,
            k=numero_autovalores,
            which="SA",
            tol=1e-10,
        )

        energias_ev = converter_joule_para_ev(autovalores)
        return energias_ev, autovetores

    def comparar_analitico_numerico(
        self,
        pontos: int = 60,
        numero: int = 4,
    ) -> Dict[str, np.ndarray]:
        """
        Compara as energias analíticas com a solução numérica por diferenças finitas.
        """
        analiticas = []
        for n in range(1, 4):
            for m in range(1, 4):
                analiticas.append(self.energia_analitica_ev(n, m))
        analiticas = np.sort(np.array(analiticas))[:numero]

        numericas, _ = self.resolver_numericamente_2d(
            pontos_y=pontos,
            pontos_z=pontos,
            numero_autovalores=numero,
        )
        erro_relativo = np.abs(numericas - analiticas) / analiticas

        return {
            "analiticas_ev": analiticas,
            "numericas_ev": numericas,
            "erro_relativo": erro_relativo,
        }

    def densidade_probabilidade(
        self,
        n: int,
        m: int,
        resolucao: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna malha (Y, Z) e |φ_{n,m}|² para visualização.
        """
        y = np.linspace(0, self.largura, resolucao)
        z = np.linspace(0, self.altura, resolucao)
        Y, Z = np.meshgrid(y, z, indexing="ij")
        phi = self.funcao_onda_analitica(n, m, Y, Z)
        densidade = np.abs(phi) ** 2
        return Y, Z, densidade

    def resumo_subbandas(self, max_n: int = 3, max_m: int = 3) -> str:
        """Gera tabela textual das primeiras sub-bandas."""
        subbandas = self.calcular_primeiras_subbandas(max_n, max_m)
        linhas = ["Índice (n,m) | Energia (eV)", "-" * 30]
        for sb in subbandas:
            linhas.append(f"  ({sb.indices[0]},{sb.indices[1]})     | {sb.energia_ev:8.4f}")
        return "\n".join(linhas)


def calcular_e11_padrao() -> float:
    """Retorna E_{1,1} do dispositivo padrão em eV."""
    calc = CalculadorSubBandas()
    return calc.energia_fundamental().energia_ev


if __name__ == "__main__":
    calc = CalculadorSubBandas()
    print(calc.resumo_subbandas())
    print(f"\nE11 = {calc.energia_fundamental().energia_ev:.4f} eV")
    comparacao = calc.comparar_analitico_numerico()
    print("\nComparação analítico vs numérico:")
    print(f"  Analíticas: {comparacao['analiticas_ev']}")
    print(f"  Numéricas : {comparacao['numericas_ev']}")
    print(f"  Erro rel. : {comparacao['erro_relativo']}")
