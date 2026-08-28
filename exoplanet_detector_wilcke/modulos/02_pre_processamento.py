"""
Módulo 02 – Pré-processamento da Curva de Luz
Inspirado em: Capítulos 1, 2 e 29 (preparação de dados funcionais)
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import median_filter


def normalizar_fluxo(fluxo: np.ndarray, metodo: str = "mediana") -> np.ndarray:
    """
    Normaliza o fluxo para que a linha de base seja ~1.0.

    Métodos: 'mediana', 'media', 'percentil'
    """
    if metodo == "mediana":
        referencia = np.nanmedian(fluxo)
    elif metodo == "media":
        referencia = np.nanmean(fluxo)
    elif metodo == "percentil":
        referencia = np.nanpercentile(fluxo, 90)
    else:
        raise ValueError("Método desconhecido")
    return fluxo / referencia


def remover_outliers(
    tempo: np.ndarray,
    fluxo: np.ndarray,
    erro: np.ndarray,
    sigma: float = 5.0,
    janela: int = 21
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove outliers usando filtro mediano + critério sigma.
    Inspirado em técnicas de robustez de estimação não-paramétrica.
    """
    fluxo_mediano = median_filter(fluxo, size=janela, mode="nearest")
    residual = fluxo - fluxo_mediano
    desvio = np.nanstd(residual)
    mascara = np.abs(residual) < sigma * desvio
    return tempo[mascara], fluxo[mascara], erro[mascara]


def detrend_polinomial(
    tempo: np.ndarray,
    fluxo: np.ndarray,
    grau: int = 2
) -> np.ndarray:
    """
    Remove tendência de longo prazo por polinômio de baixo grau
    (alternativa simples antes de métodos kernel mais sofisticados).
    """
    coeficientes = np.polyfit(tempo, fluxo, grau)
    tendencia = np.polyval(coeficientes, tempo)
    return fluxo - tendencia + 1.0  # volta a média ~1


def pre_processar_completo(
    tempo: np.ndarray,
    fluxo: np.ndarray,
    erro: np.ndarray,
    remover_outliers_flag: bool = True,
    normalizar: bool = True,
    detrend: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline completo de pré-processamento.
    """
    tempo = np.asarray(tempo, dtype=float)
    fluxo = np.asarray(fluxo, dtype=float)
    erro = np.asarray(erro, dtype=float)

    # Remove NaNs iniciais
    mascara = np.isfinite(tempo) & np.isfinite(fluxo) & np.isfinite(erro)
    tempo, fluxo, erro = tempo[mascara], fluxo[mascara], erro[mascara]

    if remover_outliers_flag:
        tempo, fluxo, erro = remover_outliers(tempo, fluxo, erro)

    if normalizar:
        fluxo = normalizar_fluxo(fluxo)

    if detrend:
        fluxo = detrend_polinomial(tempo, fluxo, grau=2)

    # Garante erro positivo
    erro = np.maximum(erro, 1e-8)

    print(f"[02] Pré-processamento concluído. Pontos finais: {len(tempo)}")
    return tempo, fluxo, erro


if __name__ == "__main__":
    from modulos import carregar_dados  # placeholder
    print("Módulo 02 carregado com sucesso.")
