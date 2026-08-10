"""
========================================================================
DOCUMENTAÇÃO TEÓRICA E REFERÊNCIA BIBLIOGRÁFICA DO MODELO:
------------------------------------------------------------------------
Livro: Métodos Avançados em Inferência Estatística Não-Paramétrica:
       Teoria Matemática, Processos de Difusão e Estimação por Kernel
Autor: Luiz Tiago Wilcke
Ano: 2026
Capítulo de Origem: Capítulo 37 - Inferência Estatística em Modelos de
                    Difusão com Volatilidade Estocástica e Saltos Co-integrados
------------------------------------------------------------------------
Correções Implementadas:
  1. Drift baseado no retorno médio histórico (Medida Real / Risco Cambial).
  2. Simulação em Log-Preços (eliminando risco de preços negativos).
  3. Saltos estocásticos multiplicativos integrados no log-espaço.
========================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Configuração de reprodutibilidade
np.random.seed(42)

print("Baixando dados reais de câmbio (USD/BRL) do mercado financeiro...")

# Baixando dados reais recentes do par USD/BRL (BRL=X)
dados_cambio = yf.download("BRL=X", start="2022-01-01", end="2026-01-01", progress=False)

# Tratamento para garantir formato de série temporal univariada
if isinstance(dados_cambio.columns, pd.MultiIndex):
    serie_precos = dados_cambio['Close'].iloc[:, 0].dropna()
else:
    serie_precos = dados_cambio['Close'].dropna()

# Calculando os retornos logarítmicos diários
retornos_diarios = np.log(serie_precos / serie_precos.shift(1)).dropna()

print(f"Total de dias úteis analisados: {len(retornos_diarios)}")
print(f"Volatilidade amostral diária: {retornos_diarios.std():.6f}")

# 1. Identificação de Choques / Saltos Extremos (Capítulo 37)
limiar_desvios = 2.55
desvio_padrao = retornos_diarios.std()
media_retorno = retornos_diarios.mean()
mascara_saltos = np.abs(retornos_diarios - media_retorno) > (limiar_desvios * desvio_padrao)
dias_com_saltos = retornos_diarios[mascara_saltos]

print(f"Número de choques/saltos extremos detectados na série cambial: {len(dias_com_saltos)}")

# 2. Estimação da Volatilidade Realizada de Alta Frequência (Proxy para v_t)
volatilidade_realizada = retornos_diarios.rolling(window=21).std() * np.sqrt(252)  # Anualizada
volatilidade_realizada = volatilidade_realizada.dropna()

# 3. Simulação Avançada em Log-Preços (Heston com Saltos Co-integrados)
S0 = float(serie_precos.iloc[-1])
v0 = float(volatilidade_realizada.iloc[-1] ** 2)
kappa = 3.0
theta = float(volatilidade_realizada.mean() ** 2)
xi = 0.3
rho = -0.6
lambda_salto = len(dias_com_saltos) / len(retornos_diarios) * 252

# CORREÇÃO 1: Uso do drift histórico real da série para simulação de risco em medida física
drift_historico = float(retornos_diarios.mean()) * 252
mu_j = 0.0
sigma_j = 0.02

dias_futuros = 252
n_trajetorias = 1000
dt = 1.0 / 252.0

# CORREÇÃO 2 & 3: Matrizes de log-preços para evitar valores negativos e manter saltos multiplicativos
log_precos_simulados = np.zeros((dias_futuros, n_trajetorias))
log_precos_simulados[0] = np.log(S0)

variancia_simulada = np.zeros((dias_futuros, n_trajetorias))
variancia_simulada[0] = v0

for t in range(1, dias_futuros):
    Z1 = np.random.normal(0, 1, n_trajetorias)
    Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * np.random.normal(0, 1, n_trajetorias)

    # Processo de Poisson para choques geopolíticos repentinos
    ocorrencia_saltos = np.random.poisson(lambda_salto * dt, n_trajetorias)
    tamanho_saltos = np.where(ocorrencia_saltos > 0, np.random.normal(mu_j, sigma_j, n_trajetorias), 0.0)

    # Atualização da variância estocástica (Modelo de Heston)
    v_antigo = np.maximum(variancia_simulada[t-1], 1e-8)
    dv = kappa * (theta - v_antigo) * dt + xi * np.sqrt(v_antigo * dt) * Z2
    variancia_simulada[t] = np.maximum(v_antigo + dv, 1e-8)

    # Incremento do log-preço (incluindo o termo de correção de convexidade -0.5*v_t*dt e saltos no log)
    d_log_s = (drift_historico - 0.5 * v_antigo) * dt + np.sqrt(v_antigo * dt) * Z1 + tamanho_saltos
    log_precos_simulados[t] = log_precos_simulados[t-1] + d_log_s

# Conversão final dos log-preços simulados de volta para o nível real via exponencial
precos_simulados = np.exp(log_precos_simulados)

# 4. Plotagem dos Resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(serie_precos.index, serie_precos.values, color='darkblue', lw=1.5, label='Taxa de Câmbio Real (USD/BRL)')
plt.scatter(dias_com_saltos.index, serie_precos.loc[dias_com_saltos.index], color='red', s=25, zorder=5, label='Choques Geopolíticos / Saltos')
plt.title('Série Histórica e Identificação de Choques Cambiais', fontsize=11, fontweight='bold')
plt.xlabel('Ano')
plt.ylabel('Taxa de Câmbio (R$)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(dias_futuros), precos_simulados, color='gray', alpha=0.03)
plt.plot(range(dias_futuros), np.mean(precos_simulados, axis=1), color='crimson', lw=2.5, label='Cenário Médio Projetado (Heston + Saltos em Log)')
plt.title('Simulação de Risco Cambial em Log-Preços (1 Ano à Frente)', fontsize=11, fontweight='bold')
plt.xlabel('Dias Úteis Futuros')
plt.ylabel('Preço Simulado (R$)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nModelo de Volatilidade Estocástica com Saltos (Capítulo 37) executado com rigor matemático em log-preços!")
