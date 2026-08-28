import matplotlib
matplotlib.use("Agg")
"""
Módulo 20 – Visualização de Resultados e Experimentos Numéricos
Inspirado em: Capítulo 29 – Aplicações práticas e experimentos numéricos
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CAMINHO_FIGURAS = Path(__file__).resolve().parent.parent / "figuras"
CAMINHO_FIGURAS.mkdir(exist_ok=True)


def plotar_curva_luz(
    tempo: np.ndarray,
    fluxo: np.ndarray,
    fluxo_suavizado: np.ndarray | None = None,
    transitos: dict | None = None,
    titulo: str = "Curva de Luz – Detecção de Exoplanetas"
) -> str:
    """
    Gera figura completa da curva de luz com suavização e candidatos a trânsito.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tempo, fluxo, ".", markersize=1, alpha=0.4, color="gray", label="Dados brutos")
    if fluxo_suavizado is not None:
        ax.plot(tempo, fluxo_suavizado, "-", color="C0", linewidth=1.5, label="Nadaraya-Watson / Projeção")
    if transitos is not None and len(transitos.get("tempos", [])) > 0:
        ax.plot(transitos["tempos"], 
                fluxo_suavizado[transitos["indices"]] if fluxo_suavizado is not None else fluxo[transitos["indices"]],
                "rv", markersize=8, label="Candidatos a trânsito")

    ax.set_xlabel("Tempo (dias)")
    ax.set_ylabel("Fluxo normalizado")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, alpha=0.3)

    caminho = CAMINHO_FIGURAS / "curva_luz_deteccao.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[20] Figura salva: {caminho}")
    return str(caminho)


def plotar_densidade_fluxo(fluxo: np.ndarray, dens: np.ndarray, grid: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fluxo, bins=50, density=True, alpha=0.5, label="Histograma")
    ax.plot(grid, dens, "r-", label="Rosenblatt-Parzen")
    ax.legend()
    ax.set_title("Estimação de Densidade do Fluxo")
    caminho = CAMINHO_FIGURAS / "densidade_fluxo.png"
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(caminho)


if __name__ == "__main__":
    print("[20] Visualização carregada.")
