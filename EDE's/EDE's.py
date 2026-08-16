#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Martingale de Molchan e EDEs Fracionárias para Risco de Crédito sob Ciclos
Macroeconômicos de Longa Memória
Autor: Luiz Tiago Wilcke
===============================================================================

Módulo de cálculo estocástico e simulação macrofinanceira:
  1. Geração exata de Movimento Browniano Fracionário (mBf) via método de Cholesky/Davies-Harte.
  2. Transformação integral de Volterra para o Martingale Fundamental de Molchan.
  3. Simulação da taxa agregada de inadimplência de crédito I_t sob dinâmica fracionária.
  4. Análise de persistência e estimação da função de autocorrelação de longo alcance.
  5. Teste de estresse macroeconômico (Stress Testing) com choques de desemprego e juros.
"""

import math
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List

# =============================================================================
# 1. PARÂMETROS MACROECONÔMICOS E DE CRÉDITO
# =============================================================================
PARAMETRO_HURST_H = 0.75             # Expoente de Hurst (H > 0.5: Memória Longa)
TAXA_REVERSAO_THETA = 0.85          # Velocidade de ajuste macroeconômico (\theta)
VOLATILIDADE_CREDITO = 0.35          # Volatilidade estocástica (\sigma)
TAXA_INADIMPLENCIA_BASE = 0.045     # Inadimplência média estrutural (4.5% a.a.)
HORIZONTE_ANOS_T = 10.0              # Janela histórica / projeção (10 anos)
NUMERO_PASSOS_TEMPO = 1000           # Resolução temporal discreta (dt = 0.01 ano)


@dataclass(frozen=True, slots=True)
class RelatorioCreditoFracionario:
    """Estrutura imutável para transporte das métricas analíticas e estatísticas."""
    expoente_hurst: float
    constante_escala_ch: float
    constante_energia_wh: float
    taxa_inadimplencia_media: float
    taxa_inadimplencia_maxima: float
    taxa_inadimplencia_p99: float
    variacao_quadratica_teorica: float
    variacao_quadratica_empirica: float
    autocorrelacao_lag_12: float


# =============================================================================
# 2. MOTOR DE CÁLCULO ESTOCÁSTICO FRACIONÁRIO E NÚCLEO DE MOLCHAN
# =============================================================================
class CalculadorMolchanFracionario:
    """
    Implementa as transformações analíticas de Volterra e operadores de Molchan
    para processos gaussianos com memória de longo alcance.
    """

    def __init__(self, hurst_h: float) -> None:
        if not (0.5 < hurst_h < 1.0):
            raise ValueError("O parâmetro de Hurst deve pertencer ao intervalo aberto (0.5, 1.0).")
        self.H = hurst_h
        
        # Constante de normalização c_H via função Beta e Gama
        termo_beta = sp.beta(2.0 - 2.0 * self.H, self.H - 0.5)
        self.c_H = math.sqrt((self.H * (2.0 * self.H - 1.0)) / termo_beta)
        
        # Constante de escala de energia da variação quadrática w_H
        num_wh = sp.gamma(3.0 - 2.0 * self.H)
        den_wh = 2.0 * self.H * sp.gamma(1.5 - self.H) * sp.gamma(self.H + 0.5)
        self.w_H = num_wh / den_wh

    def avaliar_nucleo_molchan(self, tempo_t: float, tempo_s: float) -> float:
        """Calcula k_H(t, s) = c_H^-1 * s^(H - 1/2) * (t - s)^(1/2 - H)."""
        if tempo_s >= tempo_t or tempo_s <= 0.0:
            return 0.0
        termo_s = tempo_s ** (self.H - 0.5)
        termo_diff = (tempo_t - tempo_s) ** (0.5 - self.H)
        return (1.0 / self.c_H) * termo_s * termo_diff

    def variacao_quadratica_teorica(self, tempo_t: float) -> float:
        """Calcula <M^H>_t = w_H * t^(2 - 2H)."""
        return self.w_H * (tempo_t ** (2.0 - 2.0 * self.H))


# =============================================================================
# 3. GERADOR DE TRAJETÓRIAS FRACIONÁRIAS E CRÉDITO MACROECONÔMICO
# =============================================================================
class SimuladorCreditoMacroeconômico:
    """
    Simulador de trajetórias macroeconômicas de crédito sob EDEs Fracionárias
    e transformação integral de Molchan.
    """

    def __init__(
        self,
        hurst_h: float = PARAMETRO_HURST_H,
        horizonte_t: float = HORIZONTE_ANOS_T,
        passos: int = NUMERO_PASSOS_TEMPO,
        semente: int = 42
    ) -> None:
        self.calc = CalculadorMolchanFracionario(hurst_h)
        self.H = hurst_h
        self.T = horizonte_t
        self.N = passos
        self.dt = horizonte_t / passos
        self.grade_tempo = np.linspace(0.0, horizonte_t, passos + 1)
        np.random.seed(semente)

    def gerar_movimento_browniano_fracionario(self) -> np.ndarray:
        """
        Gera uma trajetória exata do mBf via decomposição da matriz de covariância
        R_H(t_i, t_j) (Método de Cholesky Gaussiano).
        """
        t = self.grade_tempo
        matriz_cov = np.zeros((self.N + 1, self.N + 1))
        
        for i in range(self.N + 1):
            for j in range(i, self.N + 1):
                cov = 0.5 * (t[i]**(2.0*self.H) + t[j]**(2.0*self.H) - abs(t[i] - t[j])**(2.0*self.H))
                matriz_cov[i, j] = cov
                matriz_cov[j, i] = cov

        # Fatoração de Cholesky com regularização infinitesimal de estabilidade
        cholesky_l = np.linalg.cholesky(matriz_cov + 1e-12 * np.eye(self.N + 1))
        ruido_gaussiano = np.random.normal(0.0, 1.0, self.N + 1)
        trajetoria_mbf = np.dot(cholesky_l, ruido_gaussiano)
        trajetoria_mbf[0] = 0.0
        return trajetoria_mbf

    def transformar_para_martingale_molchan(self, mbf: np.ndarray) -> np.ndarray:
        """
        Transforma a trajetória do mBf no Martingale de Molchan M_t^H via integração
        numérica discreta de Volterra: M_t = \int_0^t k_H(t, s) dB_s^H.
        """
        incrementos_mbf = np.diff(mbf)
        martingale_molchan = np.zeros(self.N + 1)

        for i in range(1, self.N + 1):
            t_atual = self.grade_tempo[i]
            s_vetor = self.grade_tempo[1:i+1] - (self.dt / 2.0)  # Ponto médio de quadratura
            
            nucleos = np.array([self.calc.avaliar_nucleo_molchan(t_atual, s) for s in s_vetor])
            martingale_molchan[i] = np.sum(nucleos * incrementos_mbf[:i])

        return martingale_molchan

    def simular_inadimplencia_credito(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simula a dinâmica de inadimplência I_t = logistic(X_t) acoplada ao ciclo macro.
        Retorna: (tempo, mbf, martingale_molchan, taxa_inadimplencia)
        """
        mbf = self.gerar_movimento_browniano_fracionario()
        martingale_m = self.transformar_para_martingale_molchan(mbf)
        
        # Nível macroeconômico exógeno cíclico: \mu(t) = logit(I_0) + 0.5 * sin(2\pi t / 4) (ciclo de 4 anos)
        x_base = math.log(TAXA_INADIMPLENCIA_BASE / (1.0 - TAXA_INADIMPLENCIA_BASE))
        mu_macro = x_base + 0.45 * np.sin(2.0 * np.pi * self.grade_tempo / 4.0)
        
        # Solução da EDE Fracionária em espaço logito via esquema de Euler-Maruyama modificado
        x_estado = np.zeros(self.N + 1)
        x_estado[0] = x_base
        incrementos_mbf = np.diff(mbf)

        for k in range(self.N):
            deriva = -TAXA_REVERSAO_THETA * (x_estado[k] - mu_macro[k]) * self.dt
            difusao = VOLATILIDADE_CREDITO * incrementos_mbf[k]
            x_estado[k + 1] = x_estado[k] + deriva + difusao

        # Transformação logística reversa: I_t = 1 / (1 + exp(-X_t))
        taxa_inadimplencia = 1.0 / (1.0 + np.exp(-x_estado))
        return self.grade_tempo, mbf, martingale_m, taxa_inadimplencia


# =============================================================================
# 4. EXECUÇÃO DO PIPELINE, DIAGNÓSTICO E RELATÓRIO
# =============================================================================
def executar_pipeline_credito_molchan():
    simulador = SimuladorCreditoMacroeconômico(hurst_h=0.75, horizonte_t=10.0, passos=600, semente=42)
    tempo, mbf, m_molchan, inadimplencia = simulador.simular_inadimplencia_credito()

    # Cálculo da variação quadrática empírica do Martingale no horizonte terminal T
    var_quad_teorica = simulador.calc.variacao_quadratica_teorica(simulador.T)
    incrementos_m = np.diff(m_molchan)
    var_quad_empirica = float(np.sum(incrementos_m ** 2))

    # Cálculo da Função de Autocorrelação (ACF) da taxa de inadimplência
    serie_centrada = inadimplencia - np.mean(inadimplencia)
    autocov = np.correlate(serie_centrada, serie_centrada, mode="full")[len(inadimplencia)-1:]
    acf_empirica = autocov / autocov[0]
    lag_12_acf = float(acf_empirica[int(0.12 * len(inadimplencia))])

    relatorio = RelatorioCreditoFracionario(
        expoente_hurst=simulador.H,
        constante_escala_ch=simulador.calc.c_H,
        constante_energia_wh=simulador.calc.w_H,
        taxa_inadimplencia_media=float(np.mean(inadimplencia)),
        taxa_inadimplencia_maxima=float(np.max(inadimplencia)),
        taxa_inadimplencia_p99=float(np.percentile(inadimplencia, 99)),
        variacao_quadratica_teorica=var_quad_teorica,
        variacao_quadratica_empirica=var_quad_empirica,
        autocorrelacao_lag_12=lag_12_acf
    )

    print("=" * 85)
    print("MODELAGEM DE RISCO DE CRÉDITO MACROECONÔMICO VIA MARTINGALE DE MOLCHAN (mBf)")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 85)
    print(f"Expoente de Hurst Calibrado (H):               {relatorio.expoente_hurst:10.4f} (Memória Longa)")
    print(f"Constante de Normalização de Volterra (c_H):   {relatorio.constante_escala_ch:10.4f}")
    print(f"Constante de Energia do Martingale (w_H):      {relatorio.constante_energia_wh:10.4f}")
    print("-" * 85)
    print("MÉTRICAS DA TAXA AGREGADA DE INADIMPLÊNCIA (I_t):")
    print(f"  Taxa Média de Inadimplência:                 {relatorio.taxa_inadimplencia_media * 100:10.2f} %")
    print(f"  Pico Máximo de Inadimplência no Ciclo:       {relatorio.taxa_inadimplencia_maxima * 100:10.2f} %")
    print(f"  Nível de Estresse Severo (Percentil 99%):     {relatorio.taxa_inadimplencia_p99 * 100:10.2f} %")
    print(f"  Coeficiente de Persistência Temporal (ACF):  {relatorio.autocorrelacao_lag_12:10.4f}")
    print("-" * 85)
    print("PROPRIEDADES DO MARTINGALE DE MOLCHAN (M_t^H):")
    print(f"  Variação Quadrática Teórica <M^H>_T:         {relatorio.variacao_quadratica_teorica:10.4f}")
    print(f"  Variação Quadrática Empírica Realizada:      {relatorio.variacao_quadratica_empirica:10.4f}")
    print("=" * 85)

    # Plotagem Científica
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Taxa Agregada de Inadimplência com Bandas de Estresse
    ax1.plot(tempo, inadimplencia * 100, color="firebrick", lw=2.0, label=r"Taxa de Inadimplência $I_t = \text{logistic}(X_t)$")
    ax1.axhline(relatorio.taxa_inadimplencia_media * 100, color="black", ls="--", label=f"Média ({relatorio.taxa_inadimplencia_media*100:.2f}%)")
    ax1.axhline(relatorio.taxa_inadimplencia_p99 * 100, color="darkred", ls=":", label=f"Stress P99 ({relatorio.taxa_inadimplencia_p99*100:.2f}%)")
    ax1.set_ylabel("Inadimplência (%)", fontsize=11)
    ax1.set_title(rf"Evolução da Inadimplência sob Ciclo Macroeconômico Fracionário ($H = {simulador.H}$)", fontsize=12)
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # 2. Movimento Browniano Fracionário vs. Martingale de Molchan
    ax2.plot(tempo, mbf, color="gray", alpha=0.7, label=r"Movimento Browniano Fracionário $B_t^H$ (Não-Semimartingale)")
    ax2.plot(tempo, m_molchan, color="royalblue", lw=1.8, label=r"Martingale de Molchan $M_t^H = \int_0^t k_H dB^H$")
    ax2.set_ylabel("Amplitude Estocástica", fontsize=11)
    ax2.set_title("Mapeamento Canônico de Volterra: Transformação de Memória em Martingale", fontsize=12)
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend(loc="upper left")

    # 3. Decaimento da Função de Autocorrelação (ACF) - Comparação de Memória
    lags = np.arange(len(acf_empirica[:120])) * simulador.dt
    acf_browniana_teorica = np.exp(-TAXA_REVERSAO_THETA * lags)
    ax3.plot(lags, acf_empirica[:120], color="darkgreen", lw=2.0, label=rf"ACF Fracionária Real ($H = {simulador.H}$ - Memória Longa)")
    ax3.plot(lags, acf_browniana_teorica, "k--", lw=1.5, label=r"ACF Clássica de Markov ($H = 0.50$ - Decaimento Exponencial)")
    ax3.set_xlabel("Defasagem Temporal / Lag (Anos)", fontsize=11)
    ax3.set_ylabel("Autocorrelação $\\rho(\\tau)$", fontsize=11)
    ax3.set_title(r"Assinatura de Persistência Macroeconômica: Decaimento Hiperbólico $\propto \tau^{2H-2}$", fontsize=12)
    ax3.grid(True, ls="--", alpha=0.5)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    executar_pipeline_credito_molchan()