"""
Módulo 06 – Técnicas Avançadas de Redução de Viés por Kernel
Inspirado em: Capítulo 3 – Kernel Gama Modificado e redução de viés
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.special import gamma as gamma_func


def kernel_gama_chen(x: float, h: float, b: float = 1.0) -> float:
    """
    Kernel Gama de Chen (2000) para suporte [0, ∞).
    Útil para fluxos normalizados próximos de zero (bordas de trânsito).
    """
    if x <= 0 or h <= 0:
        return 0.0
    # Forma simplificada: densidade Gama com parâmetros dependentes de h
    shape = x / h + 1
    scale = h
    # Densidade Gama
    return (x**(shape - 1) * np.exp(-x / scale)) / (gamma_func(shape) * scale**shape)


def kernel_gama_modificado(x: np.ndarray, h: float) -> np.ndarray:
    """
    Versão vetorizada / modificada inspirada em Fauzi & Maesono (2023)
    mencionada no Cap. 3.2.
    """
    x = np.asarray(x)
    resultado = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi > 0:
            resultado[i] = kernel_gama_chen(xi, h)
    # Normalização empírica
    s = resultado.sum()
    if s > 0:
        resultado /= s
    return resultado


def estimador_vies_reduzido(
    pontos: np.ndarray,
    dados: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Estimador com kernel de ordem superior (redução de viés).
    Usa combinação linear de kernels para anular momentos até ordem 3.
    """
    # Kernel de ordem 4 simples (Gaussiano corrigido)
    dados = np.asarray(dados).ravel()
    pontos = np.asarray(pontos).ravel()
    dens = np.zeros_like(pontos)

    for i, x in enumerate(pontos):
        u = (x - dados) / h
        # Kernel de ordem 4: K4(u) = (3/2 - u²/2) φ(u)  (forma clássica)
        K4 = (1.5 - 0.5 * u**2) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        dens[i] = np.mean(K4) / h

    return dens


if __name__ == "__main__":
    print("[06] Módulo de redução de viés carregado.")
