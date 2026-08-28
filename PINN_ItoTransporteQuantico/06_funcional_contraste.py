# -*- coding: utf-8 -*-
"""
Módulo 06 — Funcional de Contraste Empírico de Itô
==================================================

Implementa o funcional de contraste γ_N(b) do Capítulo 7 do livro:

    γ_N(b) = (1/(N T)) ∑_i [ ∫ b(μ_s^i)² ds − 2 ∫ b(μ_s^i) dμ_s^i ]

Este funcional é minimizado para obter o estimador de projeção da deriva
e também entra como termo de física na perda da PINN-Itô.

Autor: Luiz Tiago Wilcke
Referência: Capítulo 7 — Equações (7.1)–(7.4) do livro (2026).
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn

from processo_ito import ProcessoIto, GeradorTrajetoriasIto
from parametros_dispositivo import criar_dispositivo_padrao


class FuncionalContraste:
    """
    Calcula o contraste empírico de mínimos quadrados contínuo para a
    função de deriva de um processo de difusão observado em múltiplas
    trajetórias curtas independentes.
    """

    def __init__(self, tempo_horizonte: float = 1.0):
        self.T = tempo_horizonte

    def contraste_discreto(
        self,
        b_avaliado: np.ndarray,
        trajetoria: np.ndarray,
        dt: float,
    ) -> float:
        """
        Aproximação discreta do contraste para uma única trajetória.

        γ ≈ (1/T) [ ∑ b² Δt − 2 ∑ b Δμ ]
        """
        incrementos = np.diff(trajetoria)
        b_meio = 0.5 * (b_avaliado[:-1] + b_avaliado[1:])  # ponto médio
        termo_quadrado = np.sum(b_meio ** 2) * dt
        termo_linear = np.sum(b_meio * incrementos)
        return (termo_quadrado - 2.0 * termo_linear) / self.T

    def contraste_lote(
        self,
        funcao_b: Callable[[np.ndarray], np.ndarray],
        trajetórias: np.ndarray,
        tempos: np.ndarray,
    ) -> float:
        """
        Calcula γ_N(b) para um lote de N trajetórias.

        Parâmetros
        ----------
        funcao_b : callable
            Função que avalia b nos pontos da trajetória.
        trajetórias : ndarray (N, n_passos+1)
        tempos : ndarray (n_passos+1,)
        """
        N, n_pontos = trajetórias.shape
        dt = tempos[1] - tempos[0]
        soma = 0.0
        for i in range(N):
            mu = trajetórias[i]
            b_vals = funcao_b(mu)
            soma += self.contraste_discreto(b_vals, mu, dt)
        return soma / N

    def contraste_torch(
        self,
        b_pred: torch.Tensor,
        mu_pred: torch.Tensor,
        dt: float = 1.0 / 100.0,
    ) -> torch.Tensor:
        """
        Versão diferenciável (PyTorch) do contraste, usada diretamente
        na perda da PINN.

        Aproximação simplificada para pontos de colocação:
        γ ≈ mean(b²) − 2 mean(b * μ)
        """
        termo_quad = torch.mean(b_pred ** 2)
        termo_lin = torch.mean(b_pred * mu_pred)
        return termo_quad - 2.0 * termo_lin

    def gradiente_contraste_analitico(
        self,
        b: np.ndarray,
        trajetoria: np.ndarray,
        dt: float,
        base: np.ndarray,
    ) -> np.ndarray:
        """
        Gradiente do contraste em relação aos coeficientes de uma base
        (usado no estimador de projeção do Cap. 7).
        """
        # Implementação do gradiente para o método de projeção
        incrementos = np.diff(trajetoria)
        n_base = base.shape[0]
        grad = np.zeros(n_base)
        for k in range(n_base):
            phi_k = base[k]
            phi_meio = 0.5 * (phi_k[:-1] + phi_k[1:])
            b_meio = 0.5 * (b[:-1] + b[1:])
            d_quad = 2.0 * np.sum(b_meio * phi_meio) * dt
            d_lin = 2.0 * np.sum(phi_meio * incrementos)
            grad[k] = (d_quad - d_lin) / self.T
        return grad


class ContrasteEmpiricoPINN(nn.Module):
    """
    Módulo PyTorch que encapsula o funcional de contraste para uso
    direto como termo de perda na PINN.
    """

    def __init__(self, peso: float = 0.1):
        super().__init__()
        self.peso = peso
        self.funcional = FuncionalContraste()

    def forward(
        self,
        b_theta: torch.Tensor,
        mu_theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retorna o valor do contraste (já multiplicado pelo peso).
        """
        gamma = self.funcional.contraste_torch(b_theta, mu_theta)
        return self.peso * gamma


def demonstrar_contraste() -> None:
    """Demonstração numérica do funcional de contraste."""
    proc = ProcessoIto()
    gerador = GeradorTrajetoriasIto(proc)
    tempos, trajs = gerador.gerar_lote(n_trajetorias=20, n_passos=100, semente=123)

    # Função de deriva verdadeira (aproximada)
    def b_verdadeira(mu):
        return 0.5 - 0.8 * mu

    # Função candidata
    def b_candidata(mu):
        return 0.4 - 0.7 * mu

    func = FuncionalContraste(tempo_horizonte=1.0)
    gamma_verd = func.contraste_lote(b_verdadeira, trajs, tempos)
    gamma_cand = func.contraste_lote(b_candidata, trajs, tempos)

    print(f"γ_N(b_verdadeira) = {gamma_verd:.6f}")
    print(f"γ_N(b_candidata)  = {gamma_cand:.6f}")
    print("(O mínimo do contraste é atingido na deriva verdadeira — Cap. 7)")


if __name__ == "__main__":
    demonstrar_contraste()
