"""
Módulo 05 – Seleção Adaptativa de Largura de Banda via Método PCO
Inspirado em: Capítulo 4 – O Método PCO (Penalized Comparison to Overfitting)
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import norm


def kernel_gaussiano(u):
    return norm.pdf(u)


def criterio_pco(
    dados: np.ndarray,
    h: float,
    h_min: float,
    lambda_pen: float = 1.0
) -> float:
    """
    Critério PCO (versão univariada simplificada):

    l_PCO(h) = ||f̂_hmin - f̂_h||² - ||K_hmin - K_h||² / n + λ ||K_h||² / n

    Minimizar este critério fornece a banda adaptativa.
    """
    dados = np.asarray(dados).ravel()
    n = len(dados)
    grid = np.linspace(dados.min() - 0.5, dados.max() + 0.5, 200)

    # Estimadores
    dens_h = _estimar_densidade_grid(dados, grid, h)
    dens_hmin = _estimar_densidade_grid(dados, grid, h_min)

    # Norma L2 empírica aproximada
    dx = grid[1] - grid[0]
    norma_diff = np.sum((dens_hmin - dens_h)**2) * dx

    # Termos de kernel (Gaussiano)
    # ||K_h||² = 1/(2√π h)  (para kernel normal)
    norma_K_h = 1.0 / (2 * np.sqrt(np.pi) * h)
    norma_K_hmin = 1.0 / (2 * np.sqrt(np.pi) * h_min)
    # Aproximação de ||K_hmin - K_h||²
    norma_diff_K = abs(norma_K_hmin - norma_K_h)  # heurística

    l_pco = norma_diff - (norma_diff_K / n) + lambda_pen * (norma_K_h / n)
    return l_pco


def _estimar_densidade_grid(dados, grid, h):
    dens = np.zeros_like(grid)
    for i, x in enumerate(grid):
        u = (x - dados) / h
        dens[i] = np.mean(kernel_gaussiano(u)) / h
    return dens


def selecionar_banda_pco(
    dados: np.ndarray,
    grid_h: np.ndarray | None = None,
    lambda_pen: float = 1.0
) -> float:
    """
    Seleciona a largura de banda ótima pelo critério PCO.
    """
    dados = np.asarray(dados).ravel()
    n = len(dados)

    if grid_h is None:
        # Grade logarítmica típica
        h_silverman = 1.06 * np.std(dados) * n**(-0.2)
        grid_h = np.logspace(np.log10(h_silverman / 10), np.log10(h_silverman * 5), 25)

    h_min = grid_h.min() * 0.5  # banda de overfitting

    valores_criterio = []
    for h in grid_h:
        try:
            val = criterio_pco(dados, h, h_min, lambda_pen)
            valores_criterio.append(val)
        except Exception:
            valores_criterio.append(np.inf)

    idx_otimo = np.argmin(valores_criterio)
    h_otimo = grid_h[idx_otimo]
    print(f"[05] Banda PCO selecionada: h = {h_otimo:.6f}")
    return float(h_otimo)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    amostra = rng.normal(0, 1, 800)
    h_pco = selecionar_banda_pco(amostra)
    print(f"h_PCO = {h_pco:.4f}")
