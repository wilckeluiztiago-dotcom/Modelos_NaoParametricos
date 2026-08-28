"""
Módulo 15 – Estimação de Quantis Não-Paramétricos por Kernel
Inspirado em: Capítulo 20 – Representação de Bahadur-Ghosh e quantis suaves
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import norm


def quantil_kernel(
    dados: np.ndarray,
    p: float,
    h: float = 0.1
) -> float:
    """
    Estimador suave de quantil (inversão da KDFE).
    """
    dados = np.asarray(dados).ravel()
    # Grade
    grid = np.linspace(dados.min() - h, dados.max() + h, 500)
    F_hat = np.array([np.mean(norm.cdf((x - dados) / h)) for x in grid])
    # Encontra o menor x tal que F_hat(x) >= p
    idx = np.searchsorted(F_hat, p)
    idx = min(idx, len(grid) - 1)
    return float(grid[idx])


def representacao_bahadur(dados: np.ndarray, p: float, h: float = 0.1) -> float:
    """
    Aproximação linear de Bahadur-Ghosh (Cap. 20.2).
    """
    q = quantil_kernel(dados, p, h)
    # Densidade no quantil
    f_q = np.mean(norm.pdf((q - dados) / h)) / h
    if f_q < 1e-8:
        return q
    # Correção
    F_emp = np.mean(dados <= q)
    return q - (F_emp - p) / f_q


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    amostra = rng.normal(0, 1, 1000)
    print("[15] Mediana suavizada:", quantil_kernel(amostra, 0.5))
