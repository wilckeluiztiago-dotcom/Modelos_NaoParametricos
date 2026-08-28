"""
Módulo 09 – Modelagem do Ruído Estelar por Equações Diferenciais Estocásticas
Inspirado em: Capítulo 6 – EDEs e Processos de Difusão
Autor: Luiz Tiago Wilcke
"""

import numpy as np


def esquema_euler_maruyama(
    b_func,
    sigma_func,
    x0: float,
    t: np.ndarray,
    seed: int | None = None
) -> np.ndarray:
    """
    Simulação de EDE via esquema de Euler-Maruyama (Cap. 6.3):

    X_{t+Δt} = X_t + b(X_t) Δt + σ(X_t) √Δt · Z ,  Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    X = np.zeros(n)
    X[0] = x0
    dt = np.diff(t)

    for i in range(1, n):
        dW = rng.normal(0, np.sqrt(dt[i-1]))
        X[i] = X[i-1] + b_func(X[i-1]) * dt[i-1] + sigma_func(X[i-1]) * dW

    return X


def modelo_ou_vasicek(theta: float = 0.5, mu: float = 0.0, sigma: float = 0.01):
    """
    Processo de Ornstein-Uhlenbeck / Vasicek (Cap. 6.6):
    dX = θ(μ - X) dt + σ dW
    Útil para modelar variabilidade estelar de curto prazo.
    """
    def b(x):
        return theta * (mu - x)

    def sig(x):
        return sigma

    return b, sig


def adicionar_ruido_sde(
    tempo: np.ndarray,
    fluxo: np.ndarray,
    intensidade: float = 0.0005
) -> np.ndarray:
    """
    Adiciona realização de um processo de difusão ao fluxo (para simulação).
    """
    b, sig = modelo_ou_vasicek(theta=2.0, mu=0.0, sigma=intensidade)
    ruido = esquema_euler_maruyama(b, sig, 0.0, tempo, seed=123)
    return fluxo + ruido


if __name__ == "__main__":
    t = np.linspace(0, 10, 500)
    b, s = modelo_ou_vasicek()
    traj = esquema_euler_maruyama(b, s, 0.0, t)
    print(f"[09] Trajetória SDE – desvio final: {traj[-1]:.4f}")
