# -*- coding: utf-8 -*-
"""
Módulo 02 — Parâmetros do Dispositivo GAAFET
============================================

Define a geometria, dopagens, tensões de polarização e propriedades do
material do nanotransistor Gate-All-Around (GAAFET) de silício sub-3 nm.

Autor: Luiz Tiago Wilcke
Referência: Capítulos 6, 7 e 36 do livro (2026).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
import math
import numpy as np

from constantes_fisicas import (
    CONSTANTES,
    massa_efetiva_silicio,
    converter_nm_para_metro,
    converter_metro_para_nm,
)


@dataclass
class GeometriaCanal:
    """
    Geometria do canal do GAAFET.

    Atributos
    ---------
    comprimento_nm : float
        Comprimento do canal L (nm).
    largura_nm : float
        Largura W do nanofio (nm).
    altura_nm : float
        Altura H do nanofio (nm).
    espessura_oxido_nm : float
        Espessura do óxido de porta (HfO2) (nm).
    """
    comprimento_nm: float = 12.0
    largura_nm: float = 3.0
    altura_nm: float = 3.0
    espessura_oxido_nm: float = 1.5

    @property
    def comprimento_m(self) -> float:
        return converter_nm_para_metro(self.comprimento_nm)

    @property
    def largura_m(self) -> float:
        return converter_nm_para_metro(self.largura_nm)

    @property
    def altura_m(self) -> float:
        return converter_nm_para_metro(self.altura_nm)

    @property
    def area_secao_m2(self) -> float:
        return self.largura_m * self.altura_m

    @property
    def volume_canal_m3(self) -> float:
        return self.area_secao_m2 * self.comprimento_m

    def validar(self) -> None:
        """Verifica consistência física da geometria."""
        if self.comprimento_nm <= 0 or self.largura_nm <= 0 or self.altura_nm <= 0:
            raise ValueError("Dimensões do canal devem ser positivas.")
        if self.largura_nm > 10.0 or self.altura_nm > 10.0:
            raise ValueError("Geometria fora da escala sub-10 nm típica de GAAFET.")


@dataclass
class PolarizacaoEletrica:
    """
    Tensões de polarização do dispositivo.
    """
    tensao_fonte_v: float = 0.0
    tensao_dreno_v: float = 0.65
    tensao_porta_v: float = 0.70
    temperatura_k: float = 300.0

    @property
    def tensao_dreno_fonte(self) -> float:
        return self.tensao_dreno_v - self.tensao_fonte_v

    def potencial_linear(self, x_normalizado: np.ndarray) -> np.ndarray:
        """
        Aproximação linear do potencial eletrostático ao longo do canal.
        V(x) = V_S + (V_D - V_S) * x , com x ∈ [0,1].
        """
        return self.tensao_fonte_v + self.tensao_dreno_fonte * x_normalizado

    def potencial_com_barreira(
        self,
        x_normalizado: np.ndarray,
        altura_barreira: float = 0.15,
        posicao_barreira: float = 0.3,
        largura_barreira: float = 0.2,
    ) -> np.ndarray:
        """
        Potencial com barreira gaussiana no início do canal (modelo simples
        de potencial de porta + dopagem).
        """
        base = self.potencial_linear(x_normalizado)
        barreira = altura_barreira * np.exp(
            -((x_normalizado - posicao_barreira) ** 2) / (2.0 * largura_barreira**2)
        )
        return base + barreira


@dataclass
class PropriedadesMaterial:
    """
    Propriedades do material do canal (silício intrínseco / levemente dopado).
    """
    massa_efetiva_relativa: float = 0.26
    constante_dieletrica_si: float = 11.7
    constante_dieletrica_oxido: float = 25.0   # HfO2 aproximado
    gap_energia_ev: float = 1.12
    afinidade_eletronica_ev: float = 4.05
    densidade_estados_efetiva_con: float = 2.8e19  # cm⁻³
    mobilidade_baixa_campo: float = 1400.0       # cm²/V·s

    @property
    def massa_efetiva_kg(self) -> float:
        return massa_efetiva_silicio(self.massa_efetiva_relativa)

    def capacidade_oxido_por_area(self, espessura_oxido_m: float) -> float:
        """
        Capacitância de óxido por unidade de área C_ox = ε_ox / t_ox.
        """
        epsilon_ox = self.constante_dieletrica_oxido * CONSTANTES.epsilon_0
        return epsilon_ox / espessura_oxido_m


@dataclass
class Dopagem:
    """
    Perfis de dopagem da fonte, canal e dreno.
    """
    densidade_fonte_cm3: float = 1.0e20
    densidade_dreno_cm3: float = 1.0e20
    densidade_canal_cm3: float = 1.0e15   # levemente dopado tipo-p ou intrínseco

    def densidade_fonte_m3(self) -> float:
        return self.densidade_fonte_cm3 * 1.0e6

    def densidade_dreno_m3(self) -> float:
        return self.densidade_dreno_cm3 * 1.0e6

    def densidade_canal_m3(self) -> float:
        return self.densidade_canal_cm3 * 1.0e6


@dataclass
class ParametrosDispositivo:
    """
    Agregador completo de todos os parâmetros do nanotransistor GAAFET.
    """
    geometria: GeometriaCanal = field(default_factory=GeometriaCanal)
    polarizacao: PolarizacaoEletrica = field(default_factory=PolarizacaoEletrica)
    material: PropriedadesMaterial = field(default_factory=PropriedadesMaterial)
    dopagem: Dopagem = field(default_factory=Dopagem)
    nome_dispositivo: str = "GAAFET_Si_12nm_3x3nm"

    def validar_todos(self) -> None:
        """Valida consistência de todos os parâmetros."""
        self.geometria.validar()
        if self.polarizacao.temperatura_k <= 0:
            raise ValueError("Temperatura deve ser positiva.")
        if self.material.massa_efetiva_relativa <= 0:
            raise ValueError("Massa efetiva relativa deve ser positiva.")

    def resumo(self) -> str:
        """Gera um resumo textual dos parâmetros principais."""
        g = self.geometria
        p = self.polarizacao
        linhas = [
            f"Dispositivo: {self.nome_dispositivo}",
            f"  Canal: L = {g.comprimento_nm:.2f} nm, W = {g.largura_nm:.2f} nm, H = {g.altura_nm:.2f} nm",
            f"  Óxido: t_ox = {g.espessura_oxido_nm:.2f} nm",
            f"  Polarização: V_S = {p.tensao_fonte_v:.3f} V, V_D = {p.tensao_dreno_v:.3f} V, V_G = {p.tensao_porta_v:.3f} V",
            f"  Temperatura: {p.temperatura_k:.1f} K",
            f"  Massa efetiva: {self.material.massa_efetiva_relativa:.3f} m0",
            f"  Dopagem fonte/dreno: {self.dopagem.densidade_fonte_cm3:.2e} cm⁻³",
        ]
        return "\n".join(linhas)

    def para_dicionario(self) -> Dict:
        """Serializa todos os parâmetros para dicionário."""
        return {
            "geometria": asdict(self.geometria),
            "polarizacao": asdict(self.polarizacao),
            "material": asdict(self.material),
            "dopagem": asdict(self.dopagem),
            "nome_dispositivo": self.nome_dispositivo,
        }

    @classmethod
    def criar_padrao_sub3nm(cls) -> "ParametrosDispositivo":
        """Fábrica do dispositivo padrão usado no readme original."""
        return cls(
            geometria=GeometriaCanal(
                comprimento_nm=12.0,
                largura_nm=3.0,
                altura_nm=3.0,
                espessura_oxido_nm=1.5,
            ),
            polarizacao=PolarizacaoEletrica(
                tensao_fonte_v=0.0,
                tensao_dreno_v=0.65,
                tensao_porta_v=0.70,
                temperatura_k=300.0,
            ),
            material=PropriedadesMaterial(massa_efetiva_relativa=0.26),
            dopagem=Dopagem(),
            nome_dispositivo="GAAFET_Si_12nm_3x3nm_sub3nm",
        )

    def calcular_campo_medio(self) -> float:
        """Campo elétrico médio aproximado E = V_DS / L (V/m)."""
        return self.polarizacao.tensao_dreno_fonte / self.geometria.comprimento_m

    def calcular_energia_termica_ev(self) -> float:
        """kT em elétron-volts."""
        return CONSTANTES.energia_termica_ev(self.polarizacao.temperatura_k)


def criar_dispositivo_padrao() -> ParametrosDispositivo:
    """Função de conveniência para obter o dispositivo padrão."""
    dispositivo = ParametrosDispositivo.criar_padrao_sub3nm()
    dispositivo.validar_todos()
    return dispositivo


if __name__ == "__main__":
    disp = criar_dispositivo_padrao()
    print(disp.resumo())
    print(f"\nCampo médio: {disp.calcular_campo_medio():.3e} V/m")
    print(f"kT = {disp.calcular_energia_termica_ev():.5f} eV")
    print(f"Área da seção: {disp.geometria.area_secao_m2:.3e} m²")
