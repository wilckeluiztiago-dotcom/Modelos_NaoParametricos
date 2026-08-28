"""
Módulo 21 – Pipeline Completo de Detecção de Exoplanetas
Integra todos os 20 módulos anteriores.
Inspirado em: Capítulo 30 + unificação de todos os métodos do livro
Autor: Luiz Tiago Wilcke
"""

import sys
from pathlib import Path
import numpy as np
import json
import importlib

# Adiciona o diretório raiz ao path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

def _imp(nome):
    """Importa módulo cujo nome começa com dígito via importlib."""
    return importlib.import_module(f"modulos.{nome}")

m01 = _imp("01_carregamento_dados_nasa")
m02 = _imp("02_pre_processamento")
m03 = _imp("03_estimacao_densidade_kernel")
m04 = _imp("04_regressao_nadaraya_watson")
m05 = _imp("05_selecao_banda_pco")
m07 = _imp("07_estimador_projecao_minimos_quadrados")
m08 = _imp("08_deteccao_transitos_kernel")
m13 = _imp("13_testes_aderencia_suavizados")
m20 = _imp("20_visualizacao_resultados")


def executar_pipeline(
    nome_alvo: str = "Kepler-10",
    missao: str = "Kepler",
    salvar_resultados: bool = True
) -> dict:
    """
    Executa o pipeline completo de detecção de exoplanetas
    usando exclusivamente métodos do livro de Luiz Tiago Wilcke.
    """
    print("=" * 70)
    print("DETECTOR DE EXOPLANETAS – MÉTODOS NÃO-PARAMÉTRICOS")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    # 1. Carregamento de dados reais da NASA
    tempo, fluxo, erro = m01.carregar_curva_luz(nome_alvo, missao=missao)

    # 2. Pré-processamento
    tempo, fluxo, erro = m02.pre_processar_completo(tempo, fluxo, erro)

    # 3. Seleção de banda via PCO (sobre o fluxo)
    # Usa subamostra para PCO (mais rápido)
    fluxo_sub = fluxo[::max(1, len(fluxo)//2000)]
    h_pco = m05.selecionar_banda_pco(fluxo_sub, lambda_pen=1.0)

    # 4. Suavização por Nadaraya-Watson
    n = len(tempo)
    if n > 6000:
        idx = np.linspace(0, n-1, 6000, dtype=int)
        tempo_s = tempo[idx]
        fluxo_s = fluxo[idx]
    else:
        tempo_s, fluxo_s = tempo.copy(), fluxo.copy()

    # Escala a banda para o domínio do tempo
    amplitude_tempo = tempo_s.max() - tempo_s.min()
    h_tempo = max(h_pco * amplitude_tempo * 0.15, amplitude_tempo / 80)

    print(f"[21] Suavizando com Nadaraya-Watson (h_tempo ≈ {h_tempo:.4f})...")
    fluxo_suavizado = m04.nadaraya_watson_vetorizado(tempo_s, tempo_s, fluxo_s, h=h_tempo)

    # 5. Alternativa: projeção de mínimos quadrados
    fluxo_proj, _ = m07.estimador_projecao_mq(tempo_s, fluxo_s, m=10)

    # 6. Detecção de dips
    # Ajusta limiar de profundidade dinamicamente
    desvio = np.nanstd(fluxo_s - fluxo_suavizado)
    limiar = max(0.0005, 2.5 * desvio)
    transitos = m08.detectar_dips_kernel(
        tempo_s, fluxo_suavizado,
        profundidade_min=limiar,
        distancia_min=max(15, int(len(tempo_s)/200))
    )

    # 7. Estimativa de período
    periodo = m08.estimar_periodo_basico(transitos["tempos"]) if transitos["n_candidatos"] > 1 else None

    # 8. Teste de aderência dos resíduos
    residuos = fluxo_s - fluxo_suavizado
    teste = m13.teste_ks_suavizado(residuos)

    # 9. Visualização principal
    caminho_fig = m20.plotar_curva_luz(
        tempo_s, fluxo_s, fluxo_suavizado, transitos,
        titulo=f"Detecção de Trânsitos – {nome_alvo} (Métodos Wilcke)"
    )

    # 10. Densidade do fluxo (extra)
    try:
        grid_dens = np.linspace(fluxo_s.min(), fluxo_s.max(), 200)
        dens = m03.estimador_rosenblatt_parzen(grid_dens, fluxo_s, h=h_pco*2)
        caminho_dens = m20.plotar_densidade_fluxo(fluxo_s, dens, grid_dens)
    except Exception as e:
        caminho_dens = None
        print(f"[21] Aviso densidade: {e}")

    # 11. Resultados
    resultados = {
        "alvo": nome_alvo,
        "missao": missao,
        "n_pontos": int(n),
        "n_pontos_analise": int(len(tempo_s)),
        "h_pco": float(h_pco),
        "h_tempo_suavizacao": float(h_tempo),
        "n_candidatos_transito": int(transitos["n_candidatos"]),
        "tempos_transito": np.asarray(transitos["tempos"]).tolist() if len(transitos.get("tempos", [])) > 0 else [],
        "profundidades": np.asarray(transitos["profundidades"]).tolist() if len(transitos.get("profundidades", [])) > 0 else [],
        "periodo_estimado_dias": float(periodo) if periodo is not None else None,
        "teste_ks_estatistica": float(teste["estatistica"]),
        "teste_ks_pvalor": float(teste["p_valor"]),
        "figura_curva": caminho_fig,
        "figura_densidade": caminho_dens,
        "autor": "Luiz Tiago Wilcke",
        "base_teorica": "Métodos Avançados em Inferência Estatística Não-Paramétrica (2026)"
    }

    print("\n" + "=" * 70)
    print("RESULTADOS FINAIS")
    print("=" * 70)
    for k, v in resultados.items():
        if k not in ("tempos_transito", "profundidades"):
            print(f"  {k}: {v}")
    if resultados["n_candidatos_transito"] > 0:
        print(f"  primeiros tempos de trânsito: {resultados['tempos_transito'][:5]}")
        print(f"  profundidades correspondentes: {resultados['profundidades'][:5]}")
    print("=" * 70)

    if salvar_resultados:
        pasta_res = RAIZ / "resultados"
        pasta_res.mkdir(exist_ok=True)
        caminho_json = pasta_res / f"resultado_{nome_alvo.replace(' ', '_')}.json"
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print(f"Resultados salvos em: {caminho_json}")

        # Salva também arrays numéricos
        np.savez_compressed(
            pasta_res / f"arrays_{nome_alvo.replace(' ', '_')}.npz",
            tempo=tempo_s,
            fluxo=fluxo_s,
            fluxo_suavizado=fluxo_suavizado,
            fluxo_projecao=fluxo_proj,
            residuos=residuos
        )
        nome_limpo = nome_alvo.replace(" ", "_")
        caminho_npz = pasta_res / f"arrays_{nome_limpo}.npz"
        print(f"Arrays salvos em: {caminho_npz}")

    return resultados


if __name__ == "__main__":
    executar_pipeline("Kepler-10")
