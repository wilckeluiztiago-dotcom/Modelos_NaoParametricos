# -*- coding: utf-8 -*-
"""
Módulo 08 — Rede Neural Físico-Informada (PINN)
===============================================

Arquitetura da rede neural que aproxima simultaneamente o quase-nível de
Fermi μ_θ(x,t) e a função de deriva b_θ(x,t).

Autor: Luiz Tiago Wilcke
Referência: Capítulo 36 do livro (2026).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import math


class CamadaLinearComInicializacao(nn.Module):
    """Camada linear com inicialização de Xavier/Glorot e ativação opcional."""

    def __init__(self, dim_entrada: int, dim_saida: int, ativacao: Optional[nn.Module] = None):
        super().__init__()
        self.linear = nn.Linear(dim_entrada, dim_saida)
        self.ativacao = ativacao
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.ativacao is not None:
            out = self.ativacao(out)
        return out


class RedePINNIto(nn.Module):
    """
    Rede Neural Físico-Informada para o problema acoplado Schrödinger + Itô.

    Entrada: (x, t) normalizados em [0,1]²
    Saída:   (μ_θ, b_θ)
    """

    def __init__(
        self,
        camadas: List[int] = None,
        ativacao: str = "tanh",
        usar_skip: bool = True,
    ):
        super().__init__()
        if camadas is None:
            camadas = [2, 64, 64, 64, 64, 2]

        self.camadas_dims = camadas
        self.usar_skip = usar_skip

        if ativacao.lower() == "tanh":
            self.ativ = nn.Tanh()
        elif ativacao.lower() == "silu":
            self.ativ = nn.SiLU()
        elif ativacao.lower() == "gelu":
            self.ativ = nn.GELU()
        else:
            self.ativ = nn.Tanh()

        # Construção das camadas ocultas
        self.camadas = nn.ModuleList()
        for i in range(len(camadas) - 2):
            self.camadas.append(
                CamadaLinearComInicializacao(camadas[i], camadas[i + 1], self.ativ)
            )
        # Camada de saída (sem ativação)
        self.saida = CamadaLinearComInicializacao(camadas[-2], camadas[-1], ativacao=None)

        # Parâmetros de Fourier para encoding opcional
        self.freq_fourier = nn.Parameter(torch.randn(16) * 2.0, requires_grad=False)

    def encoding_fourier(self, x: torch.Tensor) -> torch.Tensor:
        """Encoding de Fourier simples para melhorar representatividade de altas frequências."""
        freqs = self.freq_fourier.view(1, -1)
        return torch.cat([torch.sin(freqs * x), torch.cos(freqs * x)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parâmetros
        ----------
        x, t : Tensor de shape (batch, 1)

        Retorna
        -------
        mu, b : Tensor de shape (batch, 1)
        """
        inp = torch.cat([x, t], dim=1)  # (batch, 2)

        h = inp
        for i, camada in enumerate(self.camadas):
            h_novo = camada(h)
            if self.usar_skip and i > 0 and h_novo.shape == h.shape:
                h = h + h_novo  # conexão residual
            else:
                h = h_novo

        out = self.saida(h)
        mu = out[:, 0:1]
        b = out[:, 1:2]
        return mu, b

    def predizer_mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Atalho para obter apenas μ."""
        mu, _ = self.forward(x, t)
        return mu

    def predizer_b(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Atalho para obter apenas b."""
        _, b = self.forward(x, t)
        return b

    def numero_parametros(self) -> int:
        """Conta o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def resumo(self) -> str:
        linhas = [
            f"RedePINNIto — arquitetura {self.camadas_dims}",
            f"  Parâmetros treináveis: {self.numero_parametros():,}",
            f"  Ativação: {self.ativ.__class__.__name__}",
            f"  Skip connections: {self.usar_skip}",
        ]
        return "\n".join(linhas)


class RedePINNComCondicionamento(RedePINNIto):
    """
    Variante que recebe também a tensão de porta como entrada adicional
    (útil para sweeps de V_G).
    """

    def __init__(self, camadas: List[int] = None, **kwargs):
        if camadas is None:
            camadas = [3, 64, 64, 64, 2]
        super().__init__(camadas=camadas, **kwargs)

    def forward(self, x: torch.Tensor, t: torch.Tensor, vg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, t, vg], dim=1)
        h = inp
        for i, camada in enumerate(self.camadas):
            h_novo = camada(h)
            if self.usar_skip and i > 0 and h_novo.shape == h.shape:
                h = h + h_novo
            else:
                h = h_novo
        out = self.saida(h)
        return out[:, 0:1], out[:, 1:2]


def criar_rede_padrao(dispositivo: str = "cpu") -> RedePINNIto:
    """Fábrica da rede padrão usada nos experimentos."""
    rede = RedePINNIto(camadas=[2, 64, 64, 64, 64, 2], ativacao="tanh", usar_skip=True)
    return rede.to(dispositivo)


if __name__ == "__main__":
    rede = criar_rede_padrao()
    print(rede.resumo())
    x = torch.rand(32, 1)
    t = torch.rand(32, 1)
    mu, b = rede(x, t)
    print(f"Shape μ: {mu.shape}, Shape b: {b.shape}")
    print(f"μ médio: {mu.mean().item():.4f}, b médio: {b.mean().item():.4f}")
