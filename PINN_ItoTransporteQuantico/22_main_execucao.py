# -*- coding: utf-8 -*-
"""
Módulo 22 — Script Principal de Execução
========================================

Orquestra o treinamento completo da PINN-Itô, a extração das métricas
balísticas e a geração de relatórios e gráficos para o nanotransistor
GAAFET sub-3 nm.

Autor: Luiz Tiago Wilcke
Referência: Capítulos 7 e 36 do livro (2026).
"""

from __future__ import annotations
import os
import sys
import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constantes_fisicas import CONSTANTES
from parametros_dispositivo import criar_dispositivo_padrao
from subbandas_quanticas import CalculadorSubBandas
from equacao_schrodinger import ResidualSchrodinger
from funcional_contraste import ContrasteEmpiricoPINN
from rede_neural_pinn import criar_rede_padrao, RedePINNIto


def treinar_pinn(
    n_epocas: int = 2000,
    n_pontos: int = 256,
    lr: float = 1e-3,
    dispositivo: str = "cpu",
    verbose: bool = True,
) -> Tuple[RedePINNIto, list]:
    """Treina a PINN-Itô com perda multiobjetivo (Schrödinger + Itô + BC)."""
    device = torch.device(dispositivo)
    modelo = criar_rede_padrao(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    residual_sch = ResidualSchrodinger()
    contraste = ContrasteEmpiricoPINN(peso=0.1)

    historico = []
    w_quant, w_ito, w_bc = 1.0, 0.1, 10.0

    if verbose:
        print("=" * 60)
        print("Treinamento da PINN-Itô — GAAFET sub-3 nm")
        print(f"Epocas: {n_epocas} | Pontos de colocação: {n_pontos}")
        print(f"Dispositivo: {device}")
        print("=" * 60)

    for epoca in range(1, n_epocas + 1):
        x = torch.rand(n_pontos, 1, device=device)
        t = torch.rand(n_pontos, 1, device=device)

        L_sch = residual_sch.perda_schrodinger(modelo, x, t, tensao_porta=0.70)

        mu, b = modelo(x, t)
        L_ito = contraste(b, mu)

        x_s = torch.zeros(50, 1, device=device)
        t_s = torch.rand(50, 1, device=device)
        mu_s, _ = modelo(x_s, t_s)
        L_fonte = torch.mean(mu_s ** 2)

        x_d = torch.ones(50, 1, device=device)
        t_d = torch.rand(50, 1, device=device)
        mu_d, _ = modelo(x_d, t_d)
        L_dreno = torch.mean((mu_d - 0.65) ** 2)

        L_bc = L_fonte + L_dreno
        perda = w_quant * L_sch + w_ito * L_ito + w_bc * L_bc

        otimizador.zero_grad()
        perda.backward()
        otimizador.step()

        if epoca % 200 == 0 or epoca == 1:
            hist = {
                "epoca": epoca,
                "perda_total": perda.item(),
                "L_sch": L_sch.item(),
                "L_ito": L_ito.item(),
                "L_bc": L_bc.item(),
            }
            historico.append(hist)
            if verbose:
                print(
                    f"Época {epoca:5d} | Total: {perda.item():.4e} | "
                    f"Schr: {L_sch.item():.3e} | Itô: {L_ito.item():.3e} | BC: {L_bc.item():.3e}"
                )

    return modelo, historico


def extrair_metricas(modelo: RedePINNIto, n_pontos: int = 11) -> Dict[str, np.ndarray]:
    """Extrai μ(x), velocidade balística e transmissão Landauer."""
    dispositivo = criar_dispositivo_padrao()
    massa = dispositivo.material.massa_efetiva_kg
    q = CONSTANTES.carga_elementar
    L = dispositivo.geometria.comprimento_m
    calc_sub = CalculadorSubBandas(dispositivo)
    e11_ev = calc_sub.energia_fundamental().energia_ev

    x_nm = np.linspace(0.0, 12.0, n_pontos)
    x_norm = torch.tensor(x_nm / 12.0, dtype=torch.float32).view(-1, 1)
    t_fix = torch.ones_like(x_norm) * 0.5

    with torch.no_grad():
        mu_t, b_t = modelo(x_norm, t_fix)
        mu = mu_t.numpy().flatten()
        b = b_t.numpy().flatten()

    v = np.sqrt(2.0 * q * np.maximum(mu, 0.0) / massa) * 100.0
    v_1e7 = v / 1.0e7

    T = np.zeros_like(mu)
    hbar = CONSTANTES.hbar
    for i, xx in enumerate(x_nm):
        barreira = max(e11_ev + 0.70 - mu[i], 0.0)
        kappa = np.sqrt(2.0 * massa * barreira * q) / hbar
        integral = kappa * L * (1.0 - xx / 12.0)
        T[i] = np.clip(np.exp(-2.0 * integral), 0.0, 1.0)

    return {"x_nm": x_nm, "mu": mu, "b": b, "v_1e7": v_1e7, "T": T}


def imprimir_tabela(metricas: Dict[str, np.ndarray]) -> None:
    """Imprime a tabela de resultados no formato do readme original."""
    print("\n" + "=" * 78)
    print(f"{'Posição x (nm)':>14} | {'μ(x) (V)':>10} | {'Vel. Balística (10^7 cm/s)':>26} | {'T Landauer':>12}")
    print("-" * 78)
    for i in range(len(metricas["x_nm"])):
        print(
            f"{metricas['x_nm'][i]:14.2f} | "
            f"{metricas['mu'][i]:10.4f} | "
            f"{metricas['v_1e7'][i]:26.4f} | "
            f"{metricas['T'][i]:12.4f}"
        )
    print("=" * 78)


def gerar_graficos(metricas: Dict[str, np.ndarray], caminho: str = "resultados_pinn_ito_gaafet.png") -> None:
    """Gera e salva os gráficos principais."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(metricas["x_nm"], metricas["mu"], "b-o", linewidth=2, markersize=5)
    axes[0].set_xlabel("Posição x (nm)")
    axes[0].set_ylabel("Quase-Fermi μ(x) (V)")
    axes[0].set_title("Quase-nível de Fermi")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metricas["x_nm"], metricas["v_1e7"], "r-s", linewidth=2, markersize=5)
    axes[1].set_xlabel("Posição x (nm)")
    axes[1].set_ylabel("Velocidade (10⁷ cm/s)")
    axes[1].set_title("Velocidade Balística de Injeção")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(metricas["x_nm"], metricas["T"], "g-^", linewidth=2, markersize=5)
    axes[2].set_xlabel("Posição x (nm)")
    axes[2].set_ylabel("Transmissão T")
    axes[2].set_title("Coeficiente de Transmissão Landauer")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfico salvo em: {caminho}")


def main() -> None:
    """Ponto de entrada principal."""
    print("\n" + "#" * 60)
    print("#  PINN-Itô-Transporte-Quântico — Execução Completa")
    print("#  Autor: Luiz Tiago Wilcke (2026)")
    print("#" * 60)

    disp = criar_dispositivo_padrao()
    print("\n" + disp.resumo())

    calc = CalculadorSubBandas(disp)
    print(f"\nEnergia da sub-banda E11 = {calc.energia_fundamental().energia_ev:.4f} eV")

    inicio = time.time()
    modelo, historico = treinar_pinn(n_epocas=1500, n_pontos=256, verbose=True)
    tempo = time.time() - inicio
    print(f"\nTreinamento concluído em {tempo:.1f} s")

    metricas = extrair_metricas(modelo)
    imprimir_tabela(metricas)
    gerar_graficos(metricas)

    print("\nSimulação finalizada com sucesso.")
    print("Autor: Luiz Tiago Wilcke — 2026")


if __name__ == "__main__":
    main()
