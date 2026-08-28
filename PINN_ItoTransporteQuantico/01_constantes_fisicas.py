# -*- coding: utf-8 -*-
"""
Módulo 01 — Constantes Físicas Fundamentais
===========================================

Define todas as constantes físicas utilizadas no modelo PINN-Itô de transporte
quântico em nanotransistores GAAFET. Todas as unidades são do Sistema
Internacional (SI), com conversões explícitas para elétron-volts e nanômetros
quando necessário.

Autor: Luiz Tiago Wilcke
Referência: Capítulos 6, 7 e 36 do livro "Métodos Avançados em Inferência
            Estatística Não-Paramétrica" (2026).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Final
import numpy as np


# ---------------------------------------------------------------------------
# Constantes fundamentais (valores CODATA 2018 / 2022)
# ---------------------------------------------------------------------------

HBAR: Final[float] = 1.054571817e-34          # J·s  (constante de Planck reduzida)
H_PLANCK: Final[float] = 6.62607015e-34       # J·s  (constante de Planck)
M_ELETRON: Final[float] = 9.1093837015e-31    # kg   (massa do elétron em repouso)
CARGA_ELEMENTAR: Final[float] = 1.602176634e-19  # C   (carga elementar)
K_BOLTZMANN: Final[float] = 1.380649e-23      # J/K  (constante de Boltzmann)
EPSILON_0: Final[float] = 8.8541878128e-12    # F/m  (permissividade do vácuo)
MU_0: Final[float] = 1.25663706212e-6         # N·A⁻² (permeabilidade do vácuo)
C_LUZ: Final[float] = 299792458.0             # m/s  (velocidade da luz no vácuo)
AVOGADRO: Final[float] = 6.02214076e23        # mol⁻¹


# ---------------------------------------------------------------------------
# Constantes derivadas úteis em nanoeletrônica
# ---------------------------------------------------------------------------

EV_PARA_JOULE: Final[float] = CARGA_ELEMENTAR
JOULE_PARA_EV: Final[float] = 1.0 / CARGA_ELEMENTAR
NM_PARA_M: Final[float] = 1.0e-9
M_PARA_NM: Final[float] = 1.0e9
CM_PARA_M: Final[float] = 1.0e-2
M_PARA_CM: Final[float] = 1.0e2


@dataclass(frozen=True)
class ConstantesFisicas:
    """
    Container imutável de constantes físicas com métodos de conversão
    e cálculo de grandezas derivadas.

    Todos os atributos são finais e não podem ser alterados após a
    instanciação, garantindo consistência numérica em todo o pipeline.
    """

    hbar: float = HBAR
    h_planck: float = H_PLANCK
    m_eletron: float = M_ELETRON
    carga_elementar: float = CARGA_ELEMENTAR
    k_boltzmann: float = K_BOLTZMANN
    epsilon_0: float = EPSILON_0
    mu_0: float = MU_0
    c_luz: float = C_LUZ

    # Conversões
    ev_para_joule: float = EV_PARA_JOULE
    joule_para_ev: float = JOULE_PARA_EV
    nm_para_m: float = NM_PARA_M
    m_para_nm: float = M_PARA_NM

    def energia_termica(self, temperatura: float) -> float:
        """
        Calcula k_B * T em joules.

        Parâmetros
        ----------
        temperatura : float
            Temperatura em kelvin.

        Retorna
        -------
        float
            Energia térmica em joules.
        """
        return self.k_boltzmann * temperatura

    def energia_termica_ev(self, temperatura: float) -> float:
        """
        Calcula k_B * T em elétron-volts.
        """
        return self.energia_termica(temperatura) * self.joule_para_ev

    def comprimento_onda_de_broglie(self, momento: float) -> float:
        """
        Calcula o comprimento de onda de de Broglie λ = h / p.
        """
        return self.h_planck / momento

    def comprimento_onda_termica(self, massa: float, temperatura: float) -> float:
        """
        Comprimento de onda térmico de de Broglie.
        λ_th = h / sqrt(2 π m k_B T)
        """
        return self.h_planck / math.sqrt(2.0 * math.pi * massa * self.k_boltzmann * temperatura)

    def frequencia_ciclotron(self, campo_magnetico: float, massa: float) -> float:
        """
        Frequência ciclotrônica ω_c = e B / m.
        """
        return self.carga_elementar * campo_magnetico / massa

    def energia_fermi_gas_livre_3d(self, densidade: float, massa: float) -> float:
        """
        Energia de Fermi de um gás de elétrons livres 3D (em joules).
        E_F = (hbar² / 2m) * (3 π² n)^{2/3}
        """
        k_fermi = (3.0 * math.pi**2 * densidade)**(1.0 / 3.0)
        return (self.hbar**2 / (2.0 * massa)) * k_fermi**2

    def densidade_estados_2d(self, massa: float) -> float:
        """
        Densidade de estados 2D por unidade de área e energia (spin degenerado).
        g_2D = m / (π hbar²)
        """
        return massa / (math.pi * self.hbar**2)

    def densidade_estados_1d(self, massa: float, energia: float) -> float:
        """
        Densidade de estados 1D (por unidade de comprimento).
        g_1D(E) = (1/π) * sqrt(2m / (hbar² E))  (spin degenerado)
        """
        if energia <= 0.0:
            return 0.0
        return (1.0 / math.pi) * math.sqrt(2.0 * massa / (self.hbar**2 * energia))

    def velocidade_fermi(self, energia_fermi: float, massa: float) -> float:
        """
        Velocidade de Fermi v_F = sqrt(2 E_F / m).
        """
        return math.sqrt(2.0 * energia_fermi / massa)

    def mobilidade_drude(self, tempo_relaxacao: float, massa: float) -> float:
        """
        Mobilidade de Drude μ = e τ / m.
        """
        return self.carga_elementar * tempo_relaxacao / massa

    def difusividade_einstein(self, mobilidade: float, temperatura: float) -> float:
        """
        Relação de Einstein: D = μ k_B T / e.
        """
        return mobilidade * self.k_boltzmann * temperatura / self.carga_elementar

    def coeficiente_difusao_termica(self, temperatura: float, mobilidade: float) -> float:
        """
        Coeficiente de difusão térmica usado no termo de volatilidade de Itô.
        σ² = 2 k_B T μ_n / q   (em unidades consistentes)
        """
        return 2.0 * self.k_boltzmann * temperatura * mobilidade / self.carga_elementar

    def para_dicionario(self) -> Dict[str, float]:
        """Retorna todas as constantes como dicionário."""
        return {
            "hbar": self.hbar,
            "h_planck": self.h_planck,
            "m_eletron": self.m_eletron,
            "carga_elementar": self.carga_elementar,
            "k_boltzmann": self.k_boltzmann,
            "epsilon_0": self.epsilon_0,
            "mu_0": self.mu_0,
            "c_luz": self.c_luz,
            "ev_para_joule": self.ev_para_joule,
            "joule_para_ev": self.joule_para_ev,
            "nm_para_m": self.nm_para_m,
            "m_para_nm": self.m_para_nm,
        }

    def __repr__(self) -> str:
        return (
            f"ConstantesFisicas(hbar={self.hbar:.6e}, "
            f"m_eletron={self.m_eletron:.6e}, "
            f"carga={self.carga_elementar:.6e})"
        )


# Instância global padrão
CONSTANTES = ConstantesFisicas()


def obter_constantes() -> ConstantesFisicas:
    """Retorna a instância global de constantes físicas."""
    return CONSTANTES


def converter_ev_para_joule(energia_ev: float) -> float:
    """Converte energia de elétron-volts para joules."""
    return energia_ev * EV_PARA_JOULE


def converter_joule_para_ev(energia_joule: float) -> float:
    """Converte energia de joules para elétron-volts."""
    return energia_joule * JOULE_PARA_EV


def converter_nm_para_metro(comprimento_nm: float) -> float:
    """Converte nanômetros para metros."""
    return comprimento_nm * NM_PARA_M


def converter_metro_para_nm(comprimento_m: float) -> float:
    """Converte metros para nanômetros."""
    return comprimento_m * M_PARA_NM


def massa_efetiva_silicio(fracao: float = 0.26) -> float:
    """
    Retorna a massa efetiva longitudinal do elétron no silício.
    Valor padrão 0.26 m0 (direção [100]).
    """
    return fracao * M_ELETRON


def calcular_energia_cinetica(velocidade: float, massa: float) -> float:
    """Energia cinética clássica (1/2) m v²."""
    return 0.5 * massa * velocidade**2


def calcular_momento(energia: float, massa: float) -> float:
    """Momento a partir da energia cinética: p = sqrt(2 m E)."""
    return math.sqrt(2.0 * massa * energia)


# ---------------------------------------------------------------------------
# Tabelas de referência para validação numérica
# ---------------------------------------------------------------------------

TABELA_MASSAS_EFETIVAS: Dict[str, float] = {
    "silicio_longitudinal": 0.98,
    "silicio_transversal": 0.19,
    "silicio_conducao_media": 0.26,
    "gaas": 0.067,
    "inalas": 0.023,
    "ingeas": 0.041,
    "ge": 0.12,
}


def listar_massas_efetivas() -> None:
    """Imprime tabela de massas efetivas comuns em nanoeletrônica."""
    print("Massas efetivas relativas (m*/m0):")
    for material, fracao in TABELA_MASSAS_EFETIVAS.items():
        print(f"  {material:30s} : {fracao:.4f}")


if __name__ == "__main__":
    const = obter_constantes()
    print(const)
    print(f"kT a 300 K = {const.energia_termica_ev(300.0):.6f} eV")
    print(f"Massa efetiva Si (0.26) = {massa_efetiva_silicio():.6e} kg")
    listar_massas_efetivas()
