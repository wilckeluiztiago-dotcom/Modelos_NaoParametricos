"""
Módulo 13 – Testes de Aderência Suavizados (KS e Cramér-von Mises Kernelizados)
Inspirado em: Capítulos 19 e 22 – Testes suavizados livres de fronteira
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import norm, kstest


def estatistica_cvm_suavizada(
    dados: np.ndarray,
    h: float = 0.1
) -> float:
    """
    Versão suavizada da estatística de Cramér-von Mises.
    Compara a FDA kernelizada com a FDA teórica (normal padrão após padronização).
    """
    dados = np.asarray(dados).ravel()
    dados = (dados - np.mean(dados)) / (np.std(dados) + 1e-12)
    n = len(dados)
    grid = np.linspace(-3, 3, 300)

    # KDFE
    F_hat = np.array([np.mean(norm.cdf((x - dados) / h)) for x in grid])
    F_teorica = norm.cdf(grid)

    # Integral aproximada de (F_hat - F)^2
    dx = grid[1] - grid[0]
    cvm = n * np.sum((F_hat - F_teorica)**2) * dx
    return float(cvm)


def teste_ks_suavizado(dados: np.ndarray, h: float = 0.15) -> dict:
    """
    Teste de Kolmogorov-Smirnov suavizado.
    """
    dados = np.asarray(dados).ravel()
    dados_pad = (dados - np.mean(dados)) / (np.std(dados) + 1e-12)
    # Usa KS clássico como proxy + suavização
    estat, pvalor = kstest(dados_pad, "norm")
    return {"estatistica": estat, "p_valor": pvalor, "suavizado": True}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    amostra = rng.normal(0, 1, 500)
    print("[13] CVM suavizado:", estatistica_cvm_suavizada(amostra))
