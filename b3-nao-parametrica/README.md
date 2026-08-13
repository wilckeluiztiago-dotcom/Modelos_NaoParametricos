# Métodos Avançados em Inferência Estatística Não-Paramétrica  
## Aplicação à Bolsa de Valores Brasileira (B3)

**Autor:** Luiz Tiago Wilcke  

Este repositório contém a implementação prática dos métodos desenvolvidos no livro *Métodos Avançados em Inferência Estatística Não-Paramétrica* (Wilcke, 2026), aplicados a dados reais do mercado acionário brasileiro e à taxa Selic. O foco principal está na estimação não-paramétrica da função de deriva de processos de difusão, na simulação de trajetórias via esquema de Euler-Maruyama e no cálculo de medidas de risco a partir dessas trajetórias.

Os códigos foram escritos para serem executados de forma autônoma, com download automático de dados via Yahoo Finance e API do Banco Central do Brasil.

---

## 1. Equações Fundamentais

### 1.1 Equação Diferencial Estocástica de Itô

Consideramos o processo de difusão unidimensional

\[
dX_t = b(X_t)\,dt + \sigma(X_t)\,dW_t, \qquad X_0 = x_0,
\]

onde \(b:\mathbb{R}\to\mathbb{R}\) é a função de deriva (drift), \(\sigma:\mathbb{R}\to\mathbb{R}\) a volatilidade local e \(W\) um movimento browniano padrão.

As condições clássicas de existência e unicidade (Lipschitz global e crescimento linear) garantem solução forte única.

### 1.2 Modelo de Vasicek para a taxa Selic

A taxa de juros de curto prazo é modelada por

\[
dr_t = \kappa(\theta - r_t)\,dt + \sigma\,dW_t.
\]

A solução explícita é

\[
r_t = \theta + (r_0 - \theta)e^{-\kappa t} + \sigma\int_0^t e^{-\kappa(t-s)}\,dW_s,
\]

com distribuição marginal

\[
r_t \sim \mathcal{N}\left(\theta+(r_0-\theta)e^{-\kappa t},\ \frac{\sigma^2}{2\kappa}(1-e^{-2\kappa t})\right).
\]

### 1.3 Esquema de Euler-Maruyama

Dada a discretização \(t_k = k\Delta t\), \(\Delta t = T/n\),

\[
Y_{k+1} = Y_k + b(Y_k)\Delta t + \sigma(Y_k)\sqrt{\Delta t}\,Z_{k+1},
\]

onde \(Z_{k+1}\sim\mathcal{N}(0,1)\) i.i.d. A taxa de convergência forte é de ordem \(1/2\) e a fraca de ordem \(1\).

### 1.4 Estimador de Nadaraya-Watson para a deriva

A partir de \(N\) trajetórias (ou de uma trajetória discretizada de alta frequência) observamos os pares \((X_{t_k},\Delta X_{t_k})\). O estimador local é

\[
\hat{b}_h(x) = \frac{\displaystyle\sum_{k} K_h(X_{t_k}-x)\,\Delta X_{t_k}}{\displaystyle\sum_{k} K_h(X_{t_k}-x)\,\Delta t},
\]

onde \(K_h(u) = h^{-1}K(u/h)\) e \(K\) é um kernel simétrico de segunda ordem (geralmente gaussiano). Sob regularidade de Nikol’skii da densidade ocupacional, o viés é de ordem \(O(h^2)\) e a variância de ordem \(O((Nh)^{-1})\).

A normalidade assintótica pontual lê-se

\[
\sqrt{Nh}\bigl(\hat{b}_h(x) - b(x) - \mathrm{Bias}_h(x)\bigr) \xrightarrow{d} \mathcal{N}\Bigl(0,\ \frac{R(K)\sigma^2(x)}{T f(x)}\Bigr).
\]

### 1.5 Kernel de Rosenblatt-Parzen e Kernel Gama de Chen

Para densidades com suporte em \(\mathbb{R}\),

\[
\hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^n K\Bigl(\frac{x-X_i}{h}\Bigr).
\]

Para dados não-negativos (preços, volatilidades), utiliza-se o kernel gama de Chen:

\[
\hat{f}_{C,h}(x) = \frac{1}{n}\sum_{i=1}^n K_{G(x/h+1,\,h)}(X_i),
\]

com

\[
K_{G(\alpha,\beta)}(y) = \frac{y^{\alpha-1}e^{-y/\beta}}{\beta^\alpha\Gamma(\alpha)}.
\]

### 1.6 Modelo de Heston (volatilidade estocástica)

Quando a volatilidade não pode ser tratada como função determinística do preço,

\[
\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{V_t}\,S_t\,dW_t^1,\\
dV_t &= \kappa(\theta-V_t)\,dt + \xi\sqrt{V_t}\,dW_t^2,
\end{aligned}
\]

com \(d\langle W^1,W^2\rangle_t=\rho\,dt\) e condição de Feller \(2\kappa\theta>\xi^2\).

---

## 2. Estrutura do repositório

```
b3-nao-parametrica/
├── README.md                 (este arquivo)
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── download.py           # download de preços B3 e Selic
│   ├── kernels.py            # kernels e estimadores de densidade
│   ├── nadaraya_watson.py    # estimador de deriva
│   ├── simulacao.py          # Euler-Maruyama e Vasicek
│   └── main.py               # pipeline completo
├── results/                  # gráficos e tabelas gerados
└── data/                     # cache opcional de dados
```

---

## 3. Como executar

```bash
pip install -r requirements.txt
cd src
python main.py
```

O script baixa automaticamente:

- PETR4.SA, VALE3.SA, ITUB4.SA e Ibovespa (^BVSP) via Yahoo Finance;
- série diária da Selic (código SGS 11) via API do Banco Central.

Em seguida estima a deriva de PETR4 pelo Nadaraya-Watson, calibra um Vasicek na Selic, simula 2 000 trajetórias de um ano e calcula VaR e Expected Shortfall. Os gráficos são salvos em `results/`.

---

## 4. Detalhamento do código

### 4.1 download.py

Funções `baixar_acoes_b3` e `baixar_selic`. A primeira utiliza `yfinance`; a segunda consulta diretamente a API pública

```
https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados
```

e converte a taxa percentual diária para decimal.

### 4.2 kernels.py

Implementa o kernel gaussiano padrão e o kernel gama de Chen. A função de densidade de Rosenblatt-Parzen e a versão de Chen estão disponíveis para uso em retornos ou em variáveis positivas.

### 4.3 nadaraya_watson.py

Recebe uma série de preços, calcula os incrementos e avalia o estimador local sobre uma grade. A largura de banda padrão segue a regra de Silverman

\[
h = 1{,}06\,\hat{\sigma}\,n^{-1/5}.
\]

O resultado é uma função interpolada que pode ser usada diretamente no simulador.

### 4.4 simulacao.py

Contém o esquema de Euler-Maruyama genérico e a calibração clássica do Vasicek por regressão linear dos incrementos. A função de deriva estimada pelo Nadaraya-Watson é passada como `b_func`.

### 4.5 main.py

Orquestra todo o fluxo: download → estimação → simulação → cálculo de risco → geração dos quatro painéis gráficos (preços históricos, deriva estimada, densidade dos retornos e leque de trajetórias futuras).

---

## 5. Observações práticas

- O estimador de Nadaraya-Watson aqui é a versão discretizada do estimador contínuo apresentado no Capítulo 9 do livro. Para dados de alta frequência intradaily a aproximação melhora sensivelmente.
- A largura de banda pode ser refinada pelo método PCO (Penalized Comparison to Overfitting) descrito no Capítulo 4; a implementação atual usa a regra de Silverman por simplicidade.
- O VaR e o Expected Shortfall são obtidos empiricamente a partir das trajetórias simuladas, sem hipóteses paramétricas adicionais sobre a distribuição terminal.
- Para opções listadas na B3 (PETR, VALE etc.) o mesmo framework pode ser acoplado a um esquema de Heston ou a uma superfície de volatilidade local estimada por kernel.

---

## 6. Referência

Wilcke, L. T. (2026). *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel*. 

Os resultados numéricos e as figuras geradas por este código ilustram, de forma concreta, a aplicabilidade dos estimadores de projeção e de kernel a dados financeiros brasileiros.

---

Luiz Tiago Wilcke  
2026
