# Painel Institucional de Alocação e Risco: Ibovespa (^BVSP)

**Autor:** Luiz Tiago Wilcke  
**Referência Teórica:** *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel* (Wilcke, 2026, Capítulo 8, Seção 8.1).

---

## 📋 Sobre o Projeto

Este repositório implementa um modelo quantitativo avançado de gestão de risco e precificação de ativos baseado na teoria de **Equações Diferenciais Estocásticas (EDEs) de Itô-Lévy com Processo de Poisson Composto e Caudas Pesadas (Student-t)**. 

O sistema foi desenvolvido especificamente para modelar o comportamento do **Ibovespa (^BVSP)** no regime macroeconômico recente (pós-COVID, 2023–2026), capturando assimetrias de cauda, choques sistêmicos de liquidez e incertezas estruturais não explicadas por modelos tradicionais de difusão pura (como Black-Scholes).

---

## 🛠️ Funcionalidades e Painéis Gráficos

O script principal gera um dashboard institucional composto por quatro painéis analíticos principais e uma caixa de recomendações estratégicas:

1. **A. Histórico de Preços e Choques Proporcionais:** Visualização da série temporal do Ibovespa com marcação de saltos sistêmicos (positivos e negativos) com tamanho proporcional à magnitude do retorno.
2. **B. Retornos Diários e Limiar Estatístico:** Gráfico de retornos diários (%) validando o limiar de corte estatístico baseado em desvios robustos ($\pm 2.45\sigma$).
3. **C. Fan Chart Probabilístico & Spaghetti Plot:** Projeção estocástica de Monte Carlo (10.000 caminhos) com faixas de percentil intermediárias (Core 50%, 80% e 90%), mediana projetada, benchmark Black-Scholes e caminhos extremos (*spaghetti plot*).
4. **D. Distribuição Terminal (Wealth em T=250):** Histograma de densidade de probabilidade dos níveis finais do índice no horizonte de 1 ano, destacando o VaR 95% e o limiar de ruína (-20%).

---

## 📊 Apêndice Técnico e Diagnósticos Estatísticos

Além da visualização gerencial, o script executa uma bateria completa de auditoria estatística exibida no console:
* **Teste de Razão de Verossimilhança (LRT):** Compara o ajuste do modelo com saltos versus o modelo restrito de Black-Scholes, comprovando estatisticamente a superioridade do modelo de Itô-Lévy.
* **Critérios de Informação (AIC / BIC):** Penalização de complexidade para validação do número de parâmetros.
* **Intervalos de Confiança via Bootstrap (1.000 replicações):** Estimação rigorosa da incerteza associada aos parâmetros de intensidade ($\lambda$) e magnitude dos saltos ($\mu_{\text{jump}}$).

---

## 🚀 Como Executar

### Pré-requisitos
Certifique-se de ter o Python 3 instalado junto com as bibliotecas científicas necessárias:

```bash
pip install numpy pandas yfinance matplotlib scipy --break-system-packages
