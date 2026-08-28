# -*- coding: utf-8 -*-
"""
Módulo 04 — Equação de Schrödinger Efetiva Unidimensional
=========================================================

Implementa o residual da equação de Schrödinger efetiva ao longo do canal
(x) acoplada à sub-banda transversal E_{1,1}, usado como termo de física
na perda da PINN.

Autor: Luiz Tiago Wilcke
Referência: Capítulo 36 do livro (PINNs) e formulação do readme original.
"""

from __future__ import annotations
import math
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from constantes_fisicas import CONSTANTES
from parametros_dispositivo import ParametrosDispositivo, criar_dispositivo_padrao
from subbandas_quanticas import CalculadorSubBandas


class ResidualSchrodinger:
    """
    Calcula o residual da equação de Schrödinger efetiva:

        ∂u/∂t = - (ħ² / 2 m*) ∂²u/∂x² + [q V_gate(x) + E_{1,1} - u] u

    em unidades normalizadas adequadas para estabilidade do treinamento da PINN.
    """

    def __init__(
        self,
        dispositivo: Optional[ParametrosDispositivo] = None,
        usar_unidades_normalizadas: bool = True,
    ):
        self.dispositivo = dispositivo or criar_dispositivo_padrao()
        self.usar_normalizacao = usar_unidades_normalizadas

        calc_sub = CalculadorSubBandas(self.dispositivo)
        self.e11_joule = calc_sub.energia_fundamental().energia_joule
        self.e11_ev = calc_sub.energia_fundamental().energia_ev

        self.massa = self.dispositivo.material.massa_efetiva_kg
        self.hbar = CONSTANTES.hbar
        self.q = CONSTANTES.carga_elementar
        self.comprimento = self.dispositivo.geometria.comprimento_m

        # Fatores de normalização (x ∈ [0,1], t ∈ [0,1], energia em eV)
        self.fator_cinetico = (self.hbar ** 2) / (2.0 * self.massa * self.comprimento ** 2)
        self.fator_cinetico_ev = self.fator_cinetico * CONSTANTES.joule_para_ev

    def potencial_efetivo(
        self,
        x_norm: torch.Tensor,
        mu: torch.Tensor,
        tensao_porta: float,
    ) -> torch.Tensor:
        """
        Potencial efetivo sentido pelo elétron na sub-banda:
        V_eff = q V_G(x) + E11 - μ(x,t)
        """
        # Forma suave do potencial de porta (pode ser substituída por solução de Poisson)
        v_gate = tensao_porta * (1.0 - 0.25 * torch.sin(math.pi * x_norm))
        return v_gate + self.e11_ev - mu

    def residual(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        du_dt: torch.Tensor,
        d2u_dx2: torch.Tensor,
        tensao_porta: float = 0.70,
    ) -> torch.Tensor:
        """
        Calcula o residual pontual da equação de Schrödinger.

        Parâmetros
        ----------
        u : Tensor
            Aproximação da rede para o quase-nível de Fermi μ_θ(x,t).
        x, t : Tensor
            Coordenadas normalizadas.
        du_dt, d2u_dx2 : Tensor
            Derivadas calculadas por autograd.
        tensao_porta : float
            Tensão de porta em volts.

        Retorna
        -------
        Tensor
            Residual pontual (mesmo shape de u).
        """
        v_eff = self.potencial_efetivo(x, u, tensao_porta)
        # Termo cinético normalizado
        termo_cinetico = -self.fator_cinetico_ev * d2u_dx2
        # Residual: ∂u/∂t - (termo_cinetico + v_eff * u)
        residual = du_dt - (termo_cinetico + v_eff * u)
        return residual

    def perda_schrodinger(
        self,
        modelo: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        tensao_porta: float = 0.70,
    ) -> torch.Tensor:
        """
        Calcula a perda de colocação de Schrödinger para um lote de pontos.
        """
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)

        # Saída da rede (assumindo que o modelo retorna (mu, b) ou apenas mu)
        saida = modelo(x, t)
        if isinstance(saida, tuple):
            mu = saida[0]
        else:
            mu = saida

        # Derivadas de primeira e segunda ordem via autograd
        grad_mu = torch.autograd.grad(
            outputs=mu,
            inputs=[t, x],
            grad_outputs=torch.ones_like(mu),
            create_graph=True,
            retain_graph=True,
        )
        dmu_dt = grad_mu[0]
        dmu_dx = grad_mu[1]

        d2mu_dx2 = torch.autograd.grad(
            outputs=dmu_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dmu_dx),
            create_graph=True,
            retain_graph=True,
        )[0]

        res = self.residual(mu, x, t, dmu_dt, d2mu_dx2, tensao_porta)
        return torch.mean(res ** 2)

    def residual_estacionario(
        self,
        mu: torch.Tensor,
        x: torch.Tensor,
        d2mu_dx2: torch.Tensor,
        tensao_porta: float = 0.70,
    ) -> torch.Tensor:
        """
        Versão estacionária (∂/∂t = 0) útil para análises de regime permanente.
        """
        v_eff = self.potencial_efetivo(x, mu, tensao_porta)
        termo_cinetico = -self.fator_cinetico_ev * d2mu_dx2
        return termo_cinetico + v_eff * mu

    def densidade_probabilidade_1d(
        self,
        mu: np.ndarray,
        x_norm: np.ndarray,
    ) -> np.ndarray:
        """
        Aproximação da densidade de probabilidade a partir de |ψ|² ~ exp(-β μ)
        (estatística clássica de Boltzmann; útil para visualização).
        """
        beta = 1.0 / (CONSTANTES.energia_termica_ev(self.dispositivo.polarizacao.temperatura_k) + 1e-12)
        dens = np.exp(-beta * mu)
        dens /= np.trapz(dens, x_norm) + 1e-12
        return dens


class SolverSchrodingerNumerico:
    """
    Solver numérico de diferenças finitas para a equação de Schrödinger
    estacionária 1D (validação da PINN).
    """

    def __init__(self, dispositivo: Optional[ParametrosDispositivo] = None):
        self.dispositivo = dispositivo or criar_dispositivo_padrao()
        self.residual = ResidualSchrodinger(self.dispositivo)

    def resolver_estacionario(
        self,
        pontos: int = 200,
        tensao_porta: float = 0.70,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve a equação de Schrödinger estacionária por iteração de ponto fixo
        simples (apenas para validação de baixa complexidade).
        """
        x = np.linspace(0.0, 1.0, pontos)
        dx = x[1] - x[0]
        mu = self.dispositivo.polarizacao.potencial_linear(x).copy()

        for it in range(max_iter):
            # Segunda derivada por diferenças centrais
            d2mu = np.zeros_like(mu)
            d2mu[1:-1] = (mu[2:] - 2 * mu[1:-1] + mu[:-2]) / dx**2
            d2mu[0] = d2mu[1]
            d2mu[-1] = d2mu[-2]

            v_eff = (
                tensao_porta * (1.0 - 0.25 * np.sin(np.pi * x))
                + self.residual.e11_ev
                - mu
            )
            residual = -self.residual.fator_cinetico_ev * d2mu + v_eff * mu
            mu_novo = mu - 0.1 * residual  # passo fixo simples
            # Condições de contorno Dirichlet
            mu_novo[0] = self.dispositivo.polarizacao.tensao_fonte_v
            mu_novo[-1] = self.dispositivo.polarizacao.tensao_dreno_v

            erro = np.max(np.abs(mu_novo - mu))
            mu = mu_novo
            if erro < tol:
                break

        return x, mu


if __name__ == "__main__":
    res = ResidualSchrodinger()
    print(f"E11 = {res.e11_ev:.4f} eV")
    print(f"Fator cinético normalizado = {res.fator_cinetico_ev:.6e} eV")

    solver = SolverSchrodingerNumerico()
    x, mu = solver.resolver_estacionario()
    print(f"Solução numérica: μ(0)={mu[0]:.4f}, μ(L)={mu[-1]:.4f}, μ(meio)={mu[len(mu)//2]:.4f}")
