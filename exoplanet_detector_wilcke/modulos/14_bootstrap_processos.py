"""
Módulo 14 – Bootstrap para Processos Estocásticos e EDEs
Inspirado em: Capítulos 23 e 33 – Bootstrap paramétrico suavizado e de blocos
Autor: Luiz Tiago Wilcke
"""

import numpy as np


def bootstrap_blocos(
    serie: np.ndarray,
    tamanho_bloco: int = 50,
    n_replicas: int = 100,
    seed: int = 42
) -> list[np.ndarray]:
    """
    Bootstrap de blocos estacionários (Cap. 33.2).
    Preserva dependência temporal da curva de luz.
    """
    rng = np.random.default_rng(seed)
    n = len(serie)
    n_blocos = int(np.ceil(n / tamanho_bloco))
    replicas = []

    for _ in range(n_replicas):
        indices_inicio = rng.integers(0, n - tamanho_bloco + 1, size=n_blocos)
        blocos = [serie[i:i + tamanho_bloco] for i in indices_inicio]
        replica = np.concatenate(blocos)[:n]
        replicas.append(replica)

    return replicas


def bootstrap_parametrico_suavizado(
    residuos: np.ndarray,
    n_replicas: int = 50
) -> list[np.ndarray]:
    """
    Bootstrap paramétrico suavizado (Cap. 23).
    Reamostra resíduos de uma densidade kernel.
    """
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuos)
    rng = np.random.default_rng(123)
    replicas = [kde.resample(len(residuos)).ravel() for _ in range(n_replicas)]
    return replicas


if __name__ == "__main__":
    print("[14] Bootstrap de processos carregado.")
