# -*- coding: utf-8 -*-
"""
Módulo 05 — Processo de Difusão de Itô
======================================

Implementa a simulação e as propriedades do processo de Itô que governa
a evolução do quase-nível de Fermi μ_t ao longo do canal:

    dμ_t = b_0(μ_t) dt + σ(μ_t) dW_t

Autor: Luiz Tiago Wilcke
Referência: Capítulos 6 e 7 do livro (2026).
"""

from __future__ import annotations
import math
from typing import Callable, Optional, Tuple, List
import numpy as np
import torch

from constantes_fisicas import CONSTANTES
from parametros_dispositivo import ParametrosDispositivo, criar_dispositivo_padrao


class ProcessoIto:
    """
    Processo de difusão de Itô unidimensional com deriva e difusão
    estado-dependentes.
    """

    def __init__(
        self,
        dispositivo: Optional[ParametrosDispositivo] = None,
        funcao_deriva: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        coeficiente_difusao: Optional[float] = None,
    ):
        self.dispositivo = dispositivo or criar_dispositivo_padrao()
        self.temperatura = self.dispositivo.polarizacao.temperatura_k
        self.kT = CONSTANTES.energia_termica(self.temperatura)
        self.q = CONSTANTES.carga_elementar

        # Coeficiente de difusão padrão (volatilidade térmica)
        if coeficiente_difusao is None:
            # σ ≈ sqrt(2 kT μ_n / q)  — valor típico normalizado
            self.sigma = math.sqrt(2.0 * self.kT / self.q) * 0.1
        else:
            self.sigma = coeficiente_difusao

        # Função de deriva padrão (aceleração balística + relaxação)
        if funcao_deriva is None:
            self.funcao_deriva = self._deriva_padrao
        else:
            self.funcao_deriva = funcao_deriva

    def _deriva_padrao(self, mu: np.ndarray) -> np.ndarray:
        """
        Modelo simples de deriva: aceleração pelo campo + amortecimento.
        b(μ) = α * E - β * μ
        """
        campo = self.dispositivo.calcular_campo_medio()
        alpha = 1.0e-3   # mobilidade efetiva normalizada
        beta = 0.5       # taxa de relaxação
        return alpha * campo - beta * mu

    def coeficiente_difusao(self, mu: np.ndarray) -> np.ndarray:
        """Retorna σ(μ) (pode ser estado-dependente)."""
        return np.full_like(mu, self.sigma, dtype=float)

    def simular_euler_maruyama(
        self,
        mu0: float,
        tempo_total: float,
        n_passos: int,
        n_trajetorias: int = 1,
        semente: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula trajetórias pelo esquema de Euler-Maruyama (Cap. 6 do livro).

        Retorna
        -------
        tempos : ndarray shape (n_passos+1,)
        trajetórias : ndarray shape (n_trajetorias, n_passos+1)
        """
        if semente is not None:
            np.random.seed(semente)

        dt = tempo_total / n_passos
        tempos = np.linspace(0.0, tempo_total, n_passos + 1)
        traj = np.zeros((n_trajetorias, n_passos + 1))
        traj[:, 0] = mu0

        for i in range(n_passos):
            mu_atual = traj[:, i]
            b = self.funcao_deriva(mu_atual)
            sig = self.coeficiente_difusao(mu_atual)
            dW = np.random.normal(0.0, math.sqrt(dt), size=n_trajetorias)
            traj[:, i + 1] = mu_atual + b * dt + sig * dW

        return tempos, traj

    def simular_milstein(
        self,
        mu0: float,
        tempo_total: float,
        n_passos: int,
        n_trajetorias: int = 1,
        semente: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Esquema de Milstein (ordem forte 1.0) para maior precisão.
        """
        if semente is not None:
            np.random.seed(semente)

        dt = tempo_total / n_passos
        tempos = np.linspace(0.0, tempo_total, n_passos + 1)
        traj = np.zeros((n_trajetorias, n_passos + 1))
        traj[:, 0] = mu0

        # Aproximação da derivada de σ (constante → 0)
        dsig_dmu = 0.0

        for i in range(n_passos):
            mu_atual = traj[:, i]
            b = self.funcao_deriva(mu_atual)
            sig = self.coeficiente_difusao(mu_atual)
            dW = np.random.normal(0.0, math.sqrt(dt), size=n_trajetorias)
            termo_milstein = 0.5 * sig * dsig_dmu * (dW**2 - dt)
            traj[:, i + 1] = mu_atual + b * dt + sig * dW + termo_milstein

        return tempos, traj

    def variacao_quadratica(
        self,
        trajetoria: np.ndarray,
        dt: float,
    ) -> float:
        """
        Estima a variação quadrática [μ,μ]_T ≈ ∑ (Δμ)².
        Usado no Cap. 18 do livro para estimação de volatilidade local.
        """
        incrementos = np.diff(trajetoria)
        return np.sum(incrementos ** 2)

    def estimador_volatilidade_local(
        self,
        trajetoria: np.ndarray,
        dt: float,
        janela: int = 20,
    ) -> np.ndarray:
        """
        Estimador móvel da volatilidade local via variação quadrática.
        """
        n = len(trajetoria)
        vol = np.zeros(n)
        for i in range(n):
            ini = max(0, i - janela)
            fim = min(n, i + janela + 1)
            seg = trajetoria[ini:fim]
            if len(seg) > 1:
                vol[i] = np.sqrt(np.sum(np.diff(seg) ** 2) / ((len(seg) - 1) * dt))
        return vol


class GeradorTrajetoriasIto:
    """
    Gera conjuntos de trajetórias curtas independentes (paradigma do Cap. 7).
    """

    def __init__(self, processo: ProcessoIto):
        self.processo = processo

    def gerar_lote(
        self,
        n_trajetorias: int,
        n_passos: int,
        tempo_total: float = 1.0,
        mu0_min: float = 0.0,
        mu0_max: float = 0.65,
        semente: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera N trajetórias independentes com condições iniciais aleatórias.
        """
        if semente is not None:
            np.random.seed(semente)
        mu0s = np.random.uniform(mu0_min, mu0_max, size=n_trajetorias)
        todas = []
        for mu0 in mu0s:
            _, traj = self.processo.simular_euler_maruyama(
                mu0, tempo_total, n_passos, n_trajetorias=1
            )
            todas.append(traj[0])
        return np.linspace(0, tempo_total, n_passos + 1), np.array(todas)


if __name__ == "__main__":
    proc = ProcessoIto()
    t, traj = proc.simular_euler_maruyama(0.1, 1.0, 500, n_trajetorias=5, semente=42)
    print(f"Trajetórias shape: {traj.shape}")
    print(f"μ final médio: {traj[:, -1].mean():.4f}")
    print(f"Variação quadrática traj[0]: {proc.variacao_quadratica(traj[0], t[1]-t[0]):.6f}")
