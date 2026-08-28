"""
Módulo 08 – Detecção de Transitos via Estimadores Kernel
Inspirado em: Capítulos 9 e 11 – Nadaraya-Watson para função de deriva / duas bandas
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.signal import find_peaks


def detectar_dips_kernel(
    tempo: np.ndarray,
    fluxo_suavizado: np.ndarray,
    profundidade_min: float = 0.001,
    distancia_min: int = 20
) -> dict:
    """
    Detecta quedas de fluxo (possíveis trânsitos) no sinal suavizado.

    Usa o sinal invertido para encontrar picos (dips).
    """
    sinal_invertido = -fluxo_suavizado
    picos, propriedades = find_peaks(
        sinal_invertido,
        height=profundidade_min,
        distance=distancia_min,
        prominence=profundidade_min * 0.5
    )

    tempos_transito = tempo[picos]
    profundidades = -propriedades["peak_heights"]  # volta ao sinal original

    resultado = {
        "indices": picos,
        "tempos": tempos_transito,
        "profundidades": profundidades,
        "n_candidatos": len(picos)
    }
    print(f"[08] Candidatos a trânsito detectados: {len(picos)}")
    return resultado


def estimar_periodo_basico(tempos_transito: np.ndarray) -> float | None:
    """Estima período por diferenças sucessivas (método simples)."""
    if len(tempos_transito) < 2:
        return None
    diferencas = np.diff(np.sort(tempos_transito))
    # Remove outliers de diferença
    med = np.median(diferencas)
    mascara = np.abs(diferencas - med) < 0.3 * med
    if mascara.sum() == 0:
        return float(med)
    return float(np.median(diferencas[mascara]))


if __name__ == "__main__":
    print("[08] Módulo de detecção de trânsitos carregado.")
