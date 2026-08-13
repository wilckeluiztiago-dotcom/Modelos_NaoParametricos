"""
Download de dados da B3 e da taxa Selic.
Autor: Luiz Tiago Wilcke
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import requests


def baixar_acoes_b3(tickers=None, start="2020-01-01", end=None):
    """
    Baixa preços ajustados de ações e do Ibovespa.

    Parameters
    ----------
    tickers : list of str
        Lista de tickers no formato Yahoo (ex.: 'PETR4.SA').
    start, end : str
        Datas no formato YYYY-MM-DD.

    Returns
    -------
    pandas.DataFrame
        Preços de fechamento ajustados.
    """
    if tickers is None:
        tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "^BVSP"]
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if len(tickers) == 1:
        close = data["Close"].to_frame(name=tickers[0])
    else:
        close = data["Close"]

    return close.dropna(how="all")


def baixar_selic(start="2018-01-01"):
    """
    Baixa a série diária da Selic (código SGS 11) via API do BCB.

    A taxa é convertida de percentual ao dia para decimal.
    """
    data_ini = datetime.strptime(start, "%Y-%m-%d").strftime("%d/%m/%Y")
    data_fim = datetime.today().strftime("%d/%m/%Y")
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
        f"?formato=json&dataInicial={data_ini}&dataFinal={data_fim}"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["data"] = pd.to_datetime(df["data"], dayfirst=True)
        df = df.set_index("data").sort_index()
        df.columns = ["selic"]
        df["selic_diario"] = df["selic"] / 100.0
        return df
    except Exception as exc:
        print(f"Falha ao obter Selic: {exc}")
        return None
