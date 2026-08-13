"""
Pipeline completo de aplicação dos métodos não-paramétricos
à Bolsa de Valores Brasileira (B3).

Autor: Luiz Tiago Wilcke
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from download import baixar_acoes_b3, baixar_selic
from nadaraya_watson import nadaraya_watson_deriva
from kernels import estimador_densidade_kernel
from simulacao import simular_euler_maruyama, vasicek_parametros


def main():
    print("=" * 70)
    print("Aplicação – Inferência Não-Paramétrica na B3")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Dados
    # ------------------------------------------------------------------
    tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "^BVSP"]
    precos = baixar_acoes_b3(tickers, start="2021-01-01")
    print("\nÚltimos preços baixados:")
    print(precos.tail(3))

    retornos = np.log(precos / precos.shift(1)).dropna()

    # ------------------------------------------------------------------
    # 2. Deriva de PETR4 via Nadaraya-Watson
    # ------------------------------------------------------------------
    print("\nEstimando deriva de PETR4.SA ...")
    petr = precos["PETR4.SA"].dropna()
    grid_p, b_hat, h_opt = nadaraya_watson_deriva(petr)
    print(f"Largura de banda (Silverman): h = {h_opt:.4f}")

    # ------------------------------------------------------------------
    # 3. Densidade dos retornos
    # ------------------------------------------------------------------
    ret_petr = retornos["PETR4.SA"].dropna().values
    grid_r = np.linspace(ret_petr.min() * 1.1, ret_petr.max() * 1.1, 200)
    h_ret = 1.06 * np.std(ret_petr) * len(ret_petr) ** (-0.2)
    f_ret = estimador_densidade_kernel(ret_petr, grid_r, h_ret)

    # ------------------------------------------------------------------
    # 4. Selic e Vasicek
    # ------------------------------------------------------------------
    print("\nCalibrando Vasicek na Selic ...")
    selic_df = baixar_selic(start="2018-01-01")
    if selic_df is not None and len(selic_df) > 50:
        r = selic_df["selic_diario"].dropna().values
        kappa, theta, sigma_v = vasicek_parametros(r)
        print(f"κ = {kappa:.4f}, θ = {theta:.6f}, σ = {sigma_v:.6f}")
    else:
        print("Usando valores de referência para a Selic.")
        kappa, theta, sigma_v = 0.8, 0.00045, 0.00008

    # ------------------------------------------------------------------
    # 5. Simulação de trajetórias futuras
    # ------------------------------------------------------------------
    print("\nSimulando 2 000 trajetórias de 1 ano ...")
    b_interp = interp1d(grid_p, b_hat, kind="linear", fill_value="extrapolate")
    sigma_est = float(np.std(ret_petr) * np.mean(petr))

    def sigma_const(x):
        return np.full_like(x, sigma_est, dtype=float)

    T_fut = 1.0
    n_steps = 252
    n_paths = 2000
    x0 = float(petr.iloc[-1])

    paths = simular_euler_maruyama(
        b_interp, sigma_const, x0, T_fut, n_steps, n_paths, seed=42
    )

    # ------------------------------------------------------------------
    # 6. Medidas de risco
    # ------------------------------------------------------------------
    precos_finais = paths[:, -1]
    VaR_95 = np.percentile(precos_finais, 5)
    VaR_99 = np.percentile(precos_finais, 1)
    perdas = x0 - precos_finais
    ES_95 = np.mean(perdas[perdas > (x0 - VaR_95)])

    print(f"\nPreço atual PETR4 : R$ {x0:.2f}")
    print(f"VaR 95 % (1 ano)  : R$ {VaR_95:.2f}  ({100*(x0-VaR_95)/x0:.1f} %)")
    print(f"VaR 99 % (1 ano)  : R$ {VaR_99:.2f}")
    print(f"ES 95 %           : R$ {ES_95:.2f}")

    # ------------------------------------------------------------------
    # 7. Gráficos
    # ------------------------------------------------------------------
    os.makedirs("../results", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Preços históricos
    ax = axes[0, 0]
    for col in precos.columns:
        ax.plot(precos.index, precos[col], label=col.replace(".SA", ""), alpha=0.85)
    ax.set_title("Preços históricos – B3")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Deriva estimada
    ax = axes[0, 1]
    ax.plot(grid_p, b_hat, "b-", lw=1.8, label="Deriva NW")
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Deriva não-paramétrica – PETR4 (h = {h_opt:.3f})")
    ax.set_xlabel("Preço (R$)")
    ax.set_ylabel(r"$\hat{b}(x)$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Densidade dos retornos
    ax = axes[1, 0]
    ax.plot(grid_r, f_ret, "r-", lw=1.8, label="Kernel gaussiano")
    ax.hist(ret_petr, bins=55, density=True, alpha=0.35, color="gray")
    ax.set_title("Densidade dos retornos diários – PETR4")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trajetórias simuladas
    ax = axes[1, 1]
    t_fut = np.linspace(0, 1, n_steps + 1)
    for i in range(min(80, n_paths)):
        ax.plot(t_fut, paths[i], color="steelblue", alpha=0.04)
    q05 = np.percentile(paths, 5, axis=0)
    q50 = np.percentile(paths, 50, axis=0)
    q95 = np.percentile(paths, 95, axis=0)
    ax.plot(t_fut, q50, "k-", lw=1.8, label="Mediana")
    ax.plot(t_fut, q05, "r--", lw=1.3, label="5 %")
    ax.plot(t_fut, q95, "g--", lw=1.3, label="95 %")
    ax.axhline(x0, color="orange", ls=":", label="Preço atual")
    ax.set_title("Trajetórias simuladas – PETR4 (1 ano)")
    ax.set_xlabel("Tempo (anos)")
    ax.set_ylabel("Preço (R$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "aplicacao_b3.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nGráfico salvo em: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
