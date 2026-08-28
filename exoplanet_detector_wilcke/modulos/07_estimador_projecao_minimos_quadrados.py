"""
Módulo 07 – Estimador de Mínimos Quadrados de Projeção Ortogonal
Inspirado em: Capítulos 5 e 7 – Projeção em espaços de Hilbert / deriva de EDEs
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from numpy.polynomial.legendre import legvander, legval


def base_legendre(x: np.ndarray, m: int, intervalo: tuple = (-1, 1)) -> np.ndarray:
    """
    Constrói base de polinômios de Legendre ortonormais no intervalo.
    """
    a, b = intervalo
    # Mapeia para [-1, 1]
    z = 2 * (x - a) / (b - a) - 1
    # Matriz de Vandermonde de Legendre
    V = legvander(z, m - 1)
    # Normalização aproximada
    normas = np.sqrt(np.sum(V**2, axis=0) / len(x) + 1e-12)
    return V / normas


def estimador_projecao_mq(
    x: np.ndarray,
    y: np.ndarray,
    m: int = 8,
    intervalo: tuple | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimador de projeção de mínimos quadrados:

    ˆb_m(x) = Σ θ̂_j φ_j(x)

    onde θ̂ = Ψ̂⁻¹ Γ̂

    Usado para aproximar a forma do trânsito ou a função de deriva.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if intervalo is None:
        intervalo = (x.min(), x.max())

    Phi = base_legendre(x, m, intervalo)  # (n, m)

    # Matriz de Gram empírica Ψ̂ = (1/n) Φᵀ Φ
    Psi = (Phi.T @ Phi) / len(x)
    # Vetor Γ̂ = (1/n) Φᵀ y
    Gamma = (Phi.T @ y) / len(x)

    # Resolve sistema
    try:
        theta = np.linalg.solve(Psi + 1e-8 * np.eye(m), Gamma)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(Psi, Gamma, rcond=None)[0]

    # Avaliação
    b_hat = Phi @ theta
    return b_hat, theta


def risco_nao_adaptativo(theta: np.ndarray, sigma2: float = 1.0) -> float:
    """
    Limitante de risco não-adaptativo (Cap. 5.4).
    """
    return np.sum(theta**2) + sigma2 * len(theta)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    x = np.linspace(-1, 1, 300)
    y = np.sin(3 * x) + rng.normal(0, 0.1, 300)
    b_hat, theta = estimador_projecao_mq(x, y, m=6)
    print(f"[07] Projeção MQ – coeficientes: {theta.round(3)}")
