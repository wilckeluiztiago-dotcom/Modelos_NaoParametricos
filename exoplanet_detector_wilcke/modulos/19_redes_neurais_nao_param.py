"""
Módulo 19 – Redes Neurais como Estimadores Não-Paramétricos
Inspirado em: Capítulo 35 – Conexão Kernel Regression ↔ Redes Neurais
Autor: Luiz Tiago Wilcke
"""

import numpy as np


def ativacao_relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def rede_simples_forward(
    x: np.ndarray,
    pesos1: np.ndarray,
    bias1: np.ndarray,
    pesos2: np.ndarray,
    bias2: np.ndarray
) -> np.ndarray:
    """
    Rede com uma camada oculta (Cap. 35.1).
    ˆm(x) = Σ w_j σ(⟨a_j, x⟩ + b_j)
    """
    z1 = x.reshape(-1, 1) @ pesos1.reshape(1, -1) + bias1
    h = ativacao_relu(z1)
    return h @ pesos2 + bias2


def treinar_rede_simples(
    x: np.ndarray,
    y: np.ndarray,
    n_neuronios: int = 20,
    epocas: int = 200,
    lr: float = 0.01
) -> tuple:
    """
    Treinamento por gradiente estocástico simples.
    """
    rng = np.random.default_rng(42)
    n = len(x)
    pesos1 = rng.normal(0, 0.1, n_neuronios)
    bias1 = rng.normal(0, 0.1, n_neuronios)
    pesos2 = rng.normal(0, 0.1, n_neuronios)
    bias2 = 0.0

    for _ in range(epocas):
        # Forward
        pred = rede_simples_forward(x, pesos1, bias1, pesos2, bias2)
        erro = pred - y
        # Gradientes aproximados (muito simplificados)
        pesos2 -= lr * (h := ativacao_relu(x.reshape(-1,1)@pesos1.reshape(1,-1)+bias1)).T @ erro / n
        bias2 -= lr * np.mean(erro)

    return pesos1, bias1, pesos2, bias2


if __name__ == "__main__":
    print("[19] Redes neurais não-paramétricas carregadas.")
