"""
Módulo 01 – Carregamento de Dados Reais da NASA (Kepler / TESS)
Inspirado em: Capítulos 1 e 29 do livro (fundamentos + aplicações práticas)
Autor: Luiz Tiago Wilcke
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import lightkurve as lk
except ImportError:
    raise ImportError("Instale lightkurve: pip install lightkurve")

CAMINHO_DADOS = Path(__file__).resolve().parent.parent / "dados"
CAMINHO_DADOS.mkdir(exist_ok=True)


def carregar_curva_luz(
    nome_alvo: str = "Kepler-10",
    missao: str = "Kepler",
    autor: str = "Kepler",
    cadencia: str = "long",
    trimestre: int | None = None,
    usar_cache: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Baixa curva de luz real do MAST (NASA) via lightkurve.

    Parâmetros
    ----------
    nome_alvo : str
        Nome da estrela (ex: "Kepler-10", "KIC 6922244", "TIC 307210830")
    missao : str
        "Kepler", "K2" ou "TESS"
    autor : str
        Pipeline (geralmente "Kepler" ou "SPOC" para TESS)
    cadencia : str
        "long" ou "short"
    trimestre : int | None
        Quarter específico (Kepler) ou None para todos
    usar_cache : bool
        Se True, tenta reutilizar arquivos locais

    Retorna
    -------
    tempo : ndarray
        Tempo em dias (BKJD ou BTJD)
    fluxo : ndarray
        Fluxo normalizado (PDCSAP)
    erro_fluxo : ndarray
        Incerteza do fluxo
    """
    arquivo_cache = CAMINHO_DADOS / f"{nome_alvo.replace(' ', '_')}_{missao}.npz"

    if usar_cache and arquivo_cache.exists():
        print(f"[01] Carregando cache local: {arquivo_cache}")
        dados = np.load(arquivo_cache)
        return dados["tempo"], dados["fluxo"], dados["erro"]

    print(f"[01] Baixando dados reais da NASA para {nome_alvo} ({missao})...")

    try:
        if missao.lower() == "kepler":
            resultado_busca = lk.search_lightcurve(
                nome_alvo,
                author=autor,
                cadence=cadencia,
                mission="Kepler"
            )
            if trimestre is not None:
                resultado_busca = resultado_busca[resultado_busca.quarter == trimestre]
        elif missao.lower() == "tess":
            resultado_busca = lk.search_lightcurve(
                nome_alvo,
                author="SPOC",
                cadence="short",
                mission="TESS"
            )
        else:
            resultado_busca = lk.search_lightcurve(nome_alvo, mission=missao)

        if len(resultado_busca) == 0:
            raise ValueError(f"Nenhum dado encontrado para {nome_alvo}")

        # Baixa e costura todas as curvas disponíveis
        curvas = resultado_busca.download_all()
        if curvas is None or len(curvas) == 0:
            # Fallback: baixa a primeira
            curva = resultado_busca[0].download()
            tempo = curva.time.value
            fluxo = curva.flux.value
            erro = curva.flux_err.value if curva.flux_err is not None else np.ones_like(fluxo) * 0.001
        else:
            curva_costurada = curvas.stitch()
            tempo = curva_costurada.time.value
            fluxo = curva_costurada.flux.value
            erro = (curva_costurada.flux_err.value
                    if curva_costurada.flux_err is not None
                    else np.ones_like(fluxo) * 0.001)

        # Remove NaNs
        mascara_valida = np.isfinite(tempo) & np.isfinite(fluxo) & np.isfinite(erro)
        tempo = tempo[mascara_valida]
        fluxo = fluxo[mascara_valida]
        erro = erro[mascara_valida]

        # Salva cache
        np.savez_compressed(arquivo_cache, tempo=tempo, fluxo=fluxo, erro=erro)
        print(f"[01] Dados salvos em cache: {arquivo_cache}")
        print(f"[01] Pontos obtidos: {len(tempo)}")

        return tempo, fluxo, erro

    except Exception as e:
        print(f"[01] Erro ao baixar dados da NASA: {e}")
        print("[01] Gerando curva de luz sintética realista para demonstração...")
        return _gerar_curva_sintetica_realista()


def _gerar_curva_sintetica_realista(n_pontos: int = 5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera curva sintética com trânsito realista caso falhe o download."""
    rng = np.random.default_rng(42)
    tempo = np.linspace(0, 90, n_pontos)  # ~90 dias
    # Ruído estelar + instrumental
    fluxo = 1.0 + 0.0008 * np.sin(2 * np.pi * tempo / 12.5)  # variabilidade
    fluxo += rng.normal(0, 0.0003, n_pontos)

    # Adiciona trânsitos periódicos (período ~8.88 dias, profundidade ~0.5%)
    periodo = 8.88492
    profundidade = 0.005
    duracao = 0.15  # dias
    fase0 = 2.0
    for k in range(int(90 / periodo) + 1):
        t_centro = fase0 + k * periodo
        mascara = np.abs(tempo - t_centro) < duracao / 2
        fluxo[mascara] -= profundidade * np.exp(-0.5 * ((tempo[mascara] - t_centro) / (duracao / 4))**2)

    erro = np.full_like(fluxo, 0.00025)
    return tempo, fluxo, erro


def listar_alvos_recomendados() -> list[str]:
    """Lista de alvos com exoplanetas confirmados e dados públicos."""
    return [
        "Kepler-10",
        "Kepler-8",
        "KIC 757450",
        "Kepler-22",
        "KIC 8462852",  # Tabby's Star
        "TIC 307210830",
    ]


if __name__ == "__main__":
    tempo, fluxo, erro = carregar_curva_luz("Kepler-10")
    print(f"Tempo shape: {tempo.shape}, Fluxo médio: {np.nanmean(fluxo):.6f}")
