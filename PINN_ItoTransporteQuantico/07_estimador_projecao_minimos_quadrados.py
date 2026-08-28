# -*- coding: utf-8 -*-
"""
Módulo 07 — Estimador de Projeção de Mínimos Quadrados da Deriva
================================================================

Implementa o estimador de projeção \(\hat{b}_m\) do Capítulo 7 do livro:

    \(\hat{\theta}_m = \hat{\Psi}_m^{-1} \hat{Z}_m\)
    \(\hat{b}_m(x) = \hat{\theta}_m^\top \varphi(x)\)

onde \(\hat{\Psi}_m\) é a matriz de design empírica e \(\hat{Z}_m\) o vetor de dados.

Autor: Luiz Tiago Wilcke
Referência: Capítulo 7 (Equações 7.3–7.6) e Capítulo 5 do livro (2026).
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import numpy as np
from numpy.linalg import inv, pinv, cond
from scipy.special import legendre

from processo_ito import ProcessoIto, GeradorTrajetoriasIto
from funcional_contraste import FuncionalContraste


class BaseOrtonormal:
    """
    Gera bases ortonormais clássicas (Legendre, Fourier, Haar) para projeção.
    """

    def __init__(self, tipo: str = "legendre", dimensao: int = 8, dominio: Tuple[float, float] = (0.0, 1.0)):
        self.tipo = tipo.lower()
        self.m = dimensao
        self.a, self.b = dominio
        self.comprimento = self.b - self.a

    def avaliar(self, x: np.ndarray) -> np.ndarray:
        """
        Avalia os m primeiros elementos da base em x.
        Retorna matriz (m, len(x)).
        """
        x_norm = 2.0 * (x - self.a) / self.comprimento - 1.0  # mapeia para [-1,1]
        if self.tipo == "legendre":
            return self._legendre(x_norm)
        elif self.tipo == "fourier":
            return self._fourier(x)
        elif self.tipo == "polinomial":
            return self._polinomial(x_norm)
        else:
            raise ValueError(f"Base desconhecida: {self.tipo}")

    def _legendre(self, x_norm: np.ndarray) -> np.ndarray:
        mats = []
        for k in range(self.m):
            Pk = legendre(k)
            # normalização L2 em [-1,1]
            norma = np.sqrt(2.0 / (2 * k + 1))
            mats.append(Pk(x_norm) / norma)
        return np.array(mats)

    def _fourier(self, x: np.ndarray) -> np.ndarray:
        mats = [np.ones_like(x) / np.sqrt(self.comprimento)]
        for k in range(1, (self.m + 1) // 2):
            mats.append(np.sqrt(2 / self.comprimento) * np.cos(2 * np.pi * k * (x - self.a) / self.comprimento))
            if len(mats) < self.m:
                mats.append(np.sqrt(2 / self.comprimento) * np.sin(2 * np.pi * k * (x - self.a) / self.comprimento))
        return np.array(mats[:self.m])

    def _polinomial(self, x_norm: np.ndarray) -> np.ndarray:
        mats = []
        for k in range(self.m):
            mats.append(x_norm ** k)
        # ortogonalização Gram-Schmidt simplificada omitida por brevidade
        return np.array(mats)


class EstimadorProjecaoMinimosQuadrados:
    """
    Estimador de projeção de mínimos quadrados para a função de deriva
    a partir de múltiplas trajetórias curtas (paradigma do Cap. 7).
    """

    def __init__(
        self,
        dimensao_base: int = 6,
        tipo_base: str = "legendre",
        regularizacao: float = 1e-8,
    ):
        self.m = dimensao_base
        self.base = BaseOrtonormal(tipo=tipo_base, dimensao=dimensao_base)
        self.reg = regularizacao
        self.theta_hat: Optional[np.ndarray] = None
        self.Psi_hat: Optional[np.ndarray] = None
        self.Z_hat: Optional[np.ndarray] = None

    def _construir_matrizes(
        self,
        trajetórias: np.ndarray,
        tempos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrói \(\hat{\Psi}_m\) e \(\hat{Z}_m\).
        """
        N, n_pontos = trajetórias.shape
        dt = tempos[1] - tempos[0]
        T = tempos[-1] - tempos[0]

        Psi = np.zeros((self.m, self.m))
        Z = np.zeros(self.m)

        for i in range(N):
            mu = trajetórias[i]
            phi = self.base.avaliar(mu)          # (m, n_pontos)
            dmu = np.diff(mu)
            phi_meio = 0.5 * (phi[:, :-1] + phi[:, 1:])

            # Ψ ≈ (1/(N T)) ∑ ∫ φ φ^T ds
            Psi += (phi_meio @ phi_meio.T) * dt
            # Z ≈ (1/(N T)) ∑ ∫ φ dμ
            Z += phi_meio @ dmu

        Psi /= (N * T)
        Z /= (N * T)
        # Regularização de Tikhonov
        Psi += self.reg * np.eye(self.m)
        return Psi, Z

    def ajustar(
        self,
        trajetórias: np.ndarray,
        tempos: np.ndarray,
    ) -> "EstimadorProjecaoMinimosQuadrados":
        """
        Estima os coeficientes \(\hat{\theta}_m\).
        """
        self.Psi_hat, self.Z_hat = self._construir_matrizes(trajetórias, tempos)
        # Resolve o sistema linear
        try:
            self.theta_hat = inv(self.Psi_hat) @ self.Z_hat
        except np.linalg.LinAlgError:
            self.theta_hat = pinv(self.Psi_hat) @ self.Z_hat
        return self

    def predizer(self, x: np.ndarray) -> np.ndarray:
        """
        Avalia \(\hat{b}_m(x)\).
        """
        if self.theta_hat is None:
            raise RuntimeError("Estimador ainda não foi ajustado.")
        phi = self.base.avaliar(x)
        return self.theta_hat @ phi

    def risco_empirico(self, trajetórias: np.ndarray, tempos: np.ndarray) -> float:
        """
        Avalia o contraste γ_N no estimador obtido.
        """
        func = FuncionalContraste(tempo_horizonte=tempos[-1] - tempos[0])
        def b_hat(mu):
            return self.predizer(mu)
        return func.contraste_lote(b_hat, trajetórias, tempos)

    def condicao_matriz(self) -> float:
        """Número de condição de \(\hat{\Psi}_m\)."""
        if self.Psi_hat is None:
            raise RuntimeError("Matriz ainda não construída.")
        return cond(self.Psi_hat)


def demonstracao_estimador() -> None:
    """Demonstração completa do estimador de projeção."""
    proc = ProcessoIto()
    gerador = GeradorTrajetoriasIto(proc)
    tempos, trajs = gerador.gerar_lote(n_trajetorias=50, n_passos=80, semente=7)

    estimador = EstimadorProjecaoMinimosQuadrados(dimensao_base=5, tipo_base="legendre")
    estimador.ajustar(trajs, tempos)

    x_teste = np.linspace(0.0, 0.7, 20)
    b_est = estimador.predizer(x_teste)
    print("Estimador de projeção ajustado.")
    print(f"Coeficientes θ̂ = {estimador.theta_hat}")
    print(f"Cond(Ψ̂) = {estimador.condicao_matriz():.2e}")
    print(f"Risco empírico γ_N = {estimador.risco_empirico(trajs, tempos):.6f}")
    print(f"b̂(x) nos pontos de teste: {b_est[:5]} ...")


if __name__ == "__main__":
    demonstracao_estimador()
