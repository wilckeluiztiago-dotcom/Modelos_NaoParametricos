#!/usr/bin/env python3
"""
Ponto de entrada principal do Detector de Exoplanetas
Autor: Luiz Tiago Wilcke
Base: Métodos Avançados em Inferência Estatística Não-Paramétrica (2026)
"""

import argparse
import sys
from pathlib import Path

# Garante que o pacote seja encontrado
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modulos.21_pipeline_completo import executar_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Detector de Exoplanetas por Trânsito – Métodos Não-Paramétricos (Wilcke)"
    )
    parser.add_argument(
        "--alvo",
        type=str,
        default="Kepler-10",
        help="Nome do alvo (ex: Kepler-10, KIC 757450, TIC ...)"
    )
    parser.add_argument(
        "--missao",
        type=str,
        default="Kepler",
        choices=["Kepler", "TESS", "K2"],
        help="Missão NASA"
    )
    args = parser.parse_args()

    resultados = executar_pipeline(nome_alvo=args.alvo, missao=args.missao)
    return 0


if __name__ == "__main__":
    sys.exit(main())
