"""
Módulo 10 – Estimação da Função de Deriva em EDEs
Inspirado em: Capítulos 7 e 9 – Projeção e Nadaraya-Watson para deriva
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import importlib
estimador_projecao_mq = importlib.import_module("modulos.07_estimador_projecao_minimos_quadrados").estimador_projecao_mq


def estimar_deriva_por_projecao(
    tempo: np.ndarray,
    processo: np.ndarray,
    m: int = 6
) -> np.ndarray:
    """
    Estima a função de deriva b(x) a partir de trajetórias discretas
    usando o estimador de projeção de mínimos quadrados (Cap. 7).
    """
    # Aproximação de incrementos
    dt = np.diff(tempo)
    dX = np.diff(processo)
    # Taxa aproximada
    taxa = dX / dt
    x_pontos = processo[:-1]

    b_hat, _ = estimador_projecao_mq(x_pontos, taxa, m=m)
    return b_hat


def estimar_deriva_nadaraya(
    x_avaliacao: np.ndarray,
    x_dados: np.ndarray,
    y_incrementos: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Versão Nadaraya-Watson para a deriva (Cap. 9).
    """
    import importlib
nadaraya_watson_vetorizado = importlib.import_module("modulos.04_regressao_nadaraya_watson").nadaraya_watson_vetorizado
    return nadaraya_watson_vetorizado(x_avaliacao, x_dados, y_incrementos, h)


if __name__ == "__main__":
    print("[10] Módulo de estimação de deriva carregado.")
