"""
Módulo 12 – Estimação Suave da Função de Hazard (Taxa de Falha / Intensidade de Trânsito)
Inspirado em: Capítulo 15 – Estimação da Função de Hazard por Kernel
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import norm


def estimador_hazard_kernel(
    x: np.ndarray,
    dados: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Estimador clássico de Watson-Leadbetter / kernel para a hazard:

    λ̂(x) = f̂(x) / (1 - F̂(x))

    Aplicado aqui à distribuição de tempos de trânsito ou profundidades.
    """
    dados = np.asarray(dados).ravel()
    x = np.asarray(x).ravel()
    n = len(dados)

    # Densidade
    f_hat = np.array([np.mean(norm.pdf((xi - dados) / h)) / h for xi in x])

    # FDA empírica suavizada (KDFE simplificado)
    F_hat = np.array([np.mean(norm.cdf((xi - dados) / h)) for xi in x])

    # Evita divisão por zero
    sobrevivencia = np.maximum(1 - F_hat, 1e-8)
    lambda_hat = f_hat / sobrevivencia
    return lambda_hat


if __name__ == "__main__":
    print("[12] Estimador de hazard carregado.")
