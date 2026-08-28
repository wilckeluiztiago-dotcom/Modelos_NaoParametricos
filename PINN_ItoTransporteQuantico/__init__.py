# -*- coding: utf-8 -*-
"""
PINN-Itô-Transporte-Quântico
============================

Framework completo de Rede Neural Físico-Informada acoplada ao Cálculo
Estocástico de Itô para modelagem de transporte eletrônico quase-balístico
em nanotransistores GAAFET 3D sub-3 nm.

Autor: Luiz Tiago Wilcke
Ano: 2026
Base teórica: Métodos Avançados em Inferência Estatística Não-Paramétrica
              (Capítulos 7, 9, 25, 28 e 36)

Módulos disponíveis:
    01_constantes_fisicas
    02_parametros_dispositivo
    03_subbandas_quanticas
    04_equacao_schrodinger
    05_processo_ito
    06_funcional_contraste
    07_estimador_projecao_minimos_quadrados
    08_rede_neural_pinn
    09_perda_multiobjetivo
    10_treinamento_pinn_ito
    11_simulacao_trajetorias
    12_selecao_banda_pco
    13_kernel_suavizacao
    14_velocidade_balistica
    15_transmissao_landauer
    16_corrente_balistica
    17_extracao_metricas
    18_visualizacao_resultados
    19_validacao_numerica
    20_utilitarios_matematicos
    21_configuracao_experimento
    22_main_execucao
    23_testes_integracao
"""

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"
__email__ = "wilckeluiztiago@gmail.com"
__livro__ = "Métodos Avançados em Inferência Estatística Não-Paramétrica (2026)"

from . import constantes_fisicas
from . import parametros_dispositivo
from . import subbandas_quanticas
from . import equacao_schrodinger
from . import processo_ito
from . import funcional_contraste
from . import estimador_projecao_minimos_quadrados
from . import rede_neural_pinn
from . import perda_multiobjetivo
from . import treinamento_pinn_ito
from . import simulacao_trajetorias
from . import selecao_banda_pco
from . import kernel_suavizacao
from . import velocidade_balistica
from . import transmissao_landauer
from . import corrente_balistica
from . import extracao_metricas
from . import visualizacao_resultados
from . import validacao_numerica
from . import utilitarios_matematicos
from . import configuracao_experimento

__all__ = [
    "constantes_fisicas",
    "parametros_dispositivo",
    "subbandas_quanticas",
    "equacao_schrodinger",
    "processo_ito",
    "funcional_contraste",
    "estimador_projecao_minimos_quadrados",
    "rede_neural_pinn",
    "perda_multiobjetivo",
    "treinamento_pinn_ito",
    "simulacao_trajetorias",
    "selecao_banda_pco",
    "kernel_suavizacao",
    "velocidade_balistica",
    "transmissao_landauer",
    "corrente_balistica",
    "extracao_metricas",
    "visualizacao_resultados",
    "validacao_numerica",
    "utilitarios_matematicos",
    "configuracao_experimento",
]
