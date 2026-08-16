# PINN-BlackCox-CreditRisk: Avaliação Estrutural de Risco de Crédito e Covenants Financeiros via Redes Neurais Físico-Informadas

**Autor:** Luiz Tiago Wilcke  
**Área:** Engenharia Financeira, Risco de Crédito Quantitativo e Redes Neurais Físico-Informadas (PINNs)

---

## Resumo

Este repositório apresenta a formulação matemática, o embasamento analítico e a implementação computacional de uma **Rede Neural Físico-Informada (PINN - *Physics-Informed Neural Network*)** para a resolução da equação diferencial parcial de precificação estrutural de crédito sob o **Modelo de Black-Cox (1976)** com barreira de absorção dependente do tempo. O modelo captura cláusulas contratuais de proteção (*financial covenants*) e antecipação de falência corporativa (*first-passage time*), permitindo calcular de forma contínua e sem malha o valor da dívida arriscada $\Psi(V, t)$, o patrimônio líquido da firma $E(V, t)$, a taxa de retorno no vencimento (*yield*) e o *credit spread* em pontos-base.

---

## 1. Fundamentação Teórica do Modelo Estrutural de Black-Cox

### 1.1. Processo Estocástico do Valor da Firma

Sob a medida neutra ao risco $\mathbb{Q}$, o valor total dos ativos da empresa emissora $V_t$ é regido por uma Equação Diferencial Estocástica de difusão geométrica:

$$
dV_t = (r - q) V_t \, dt + \sigma_V V_t \, dW_t
$$

Onde:
* $r$ é a taxa de juros livre de risco contínua anualizada.
* $q$ é a taxa de distribuição de dividendos e desembolsos de caixa.
* $\sigma_V$ é a volatilidade dos ativos da empresa.
* $W_t$ é o processo browniano padrão unidimensional.

---

### 1.2. Barreira Estocástica de Insolvência e Covenants ($V_B(t)$)

Diferente do modelo pioneiro de Merton (1974), no qual o default só pode ocorrer na data de vencimento $T$, o modelo de Black-Cox permite a liquidação antecipada da empresa no primeiro instante em que o valor dos ativos atinge uma barreira de segurança $V_B(t)$:

$$
\tau = \inf \left\{ t \ge 0 : V_t \le V_B(t) \right\}
$$

A barreira exponencial de proteção dos credores é parametrizada por:

$$
V_B(t) = K \exp\left( -\gamma (T - t) \right)
$$

onde $K$ representa o valor de referência contratual e $\gamma$ é a taxa de decaimento temporal do covenant.

---

### 1.3. Equação Diferencial Parcial de Precificação da Dívida

Pelo argumento de não-arbitragem e replicação dinâmica de carteira, o valor da dívida corporativa $\Psi(V, t)$ satisfaz a EDP parabólica de Black-Scholes-Merton no domínio aberto $\Omega = \left\{ (V, t) : V > V_B(t), \, t \in [0, T) \right\}$:

$$
\frac{\partial \Psi}{\partial t} + (r - q) V \frac{\partial \Psi}{\partial V} + \frac{1}{2} \sigma_V^2 V^2 \frac{\partial^2 \Psi}{\partial V^2} - r \Psi = 0
$$

#### Condições de Contorno e Terminal:

1. **Condição Terminal no Vencimento ($t = T$):**
   $$
   \Psi(V, T) = \min(V, F) = F - \max(F - V, 0)
   $$
   onde $F$ é o valor nominal da dívida.

2. **Condição de Default na Barreira de Covenants ($V = V_B(t)$):**
   $$
   \Psi(V_B(t), t) = \beta V_B(t)
   $$
   onde $\beta \in [0, 1]$ é a taxa de recuperação dos credores na falência.

3. **Condição Assintótica de Solvência Plena ($V \to \infty$):**
   $$
   \lim_{V \to \infty} \Psi(V, t) = F \exp\left( -r (T - t) \right)
   $$

---

## 2. Métricas de Crédito Extraídas

A partir do valor obtido pela PINN para a dívida arriscada $\Psi(V, t)$, deduzem-se as métricas de precificação e risco de crédito:

1. **Valor do Patrimônio Líquido (*Equity*):**
   $$
   E(V, t) = V - \Psi(V, t)
   $$

2. **Taxa de Retorno até o Vencimento (*Yield to Maturity*):**
   $$
   y(V, t) = -\frac{\ln\left( \frac{\Psi(V, t)}{F} \right)}{T - t}
   $$

3. **Spread de Crédito (*Credit Spread* em bps):**
   $$
   s(V, t) = \left[ y(V, t) - r \right] \times 10^4 \quad [\text{bps}]
   $$

---

## 3. Arquitetura da Rede Neural Físico-Informada (PINN)

A rede neural aproxima diretamente o preço da dívida:

$$
\mathcal{N}_\theta : (V, t) \in \mathbb{R}^2 \longmapsto \hat{\Psi}(V, t) \in \mathbb{R}
$$

```
   (V, t)
     │
     ▼
 ┌───────────────┐
 │ Fourier Feat. │ ──> Projeção Senoidal/Cossenoidal (32 dimensões)
 └───────────────┘
     │
     ▼
 ┌───────────────┐
 │ Camada Linear │ ──> Linear(32, 64) + Tanh()
 └───────────────┘
     │
     ▼
 ┌───────────────┐
 │ 3x Hidden L.  │ ──> 3 Camadas Densas (64 neurônios, Tanh)
 └───────────────┘
     │
     ▼
 ┌───────────────┐
 │ Camada Saída  │ ──> Linear(64, 1) * F
 └───────────────┘
     │
     ▼
   \hat{\Psi}(V, t)
```

### Função de Perda Composta

$$\mathcal{L}_{\text{total}}(\theta) = w_{\text{EDP}} \mathcal{L}_{\text{EDP}}(\theta) + w_{\text{TC}} \mathcal{L}_{\text{Terminal}}(\theta) + w_{\text{barr}} \mathcal{L}_{\text{Barreira}}(\theta) + w_{\text{inf}} \mathcal{L}_{\text{Assintotico}}(\theta)$$

Onde os resíduos da EDP são computados via diferenciação automática (*Autograd*):

$$
\mathcal{L}_{\text{EDP}}(\theta) = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \frac{\partial \hat{\Psi}}{\partial t} + (r - q) V^{(i)} \frac{\partial \hat{\Psi}}{\partial V} + \frac{1}{2} \sigma_V^2 (V^{(i)})^2 \frac{\partial^2 \hat{\Psi}}{\partial V^2} - r \hat{\Psi} \right|^2
$$

---

## 4. Resultados Numéricos da Avaliação de Carteira

Resultados obtidos para uma dívida com $F = \text{R\$\ } 100,00$, $T = 5\text{ anos}$, $r = 5.0\%$, $q = 2.0\%$, $\sigma_V = 25.0\%$, $K = \text{R\$\ } 60,00$, $\gamma = 3.0\%$ e $\beta = 60.0\%$:

| Valor Firma $V$ (R$) | Dívida $\Psi$ (R$) | Equity $E$ (R$) | Yield Anual (%) | Credit Spread (bps) | Classificação |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **65.00** | 41.28 | 23.72 | 17.701% | 1270.1 | Alto Risco (Distressed) |
| **74.64** | 49.85 | 24.79 | 13.921% | 892.1 | Alto Risco (High Yield) |
| **84.29** | 57.12 | 27.17 | 11.198% | 619.8 | Grau Especulativo |
| **93.93** | 63.14 | 30.79 | 9.201% | 420.1 | Grau Especulativo |
| **103.57** | 67.89 | 35.68 | 7.745% | 274.5 | Grau Médio |
| **113.21** | 71.45 | 41.76 | 6.726% | 172.6 | Grau Médio |
| **122.86** | 73.98 | 48.88 | 6.033% | 103.3 | Grau de Investimento |
| **132.50** | 75.71 | 56.79 | 5.572% | 57.2 | Grau de Investimento |
| **142.14** | 76.84 | 65.30 | 5.277% | 27.7 | Grau de Investimento |
| **161.43** | 77.78 | 83.65 | 5.034% | 3.4 | Risco Quase Nulo |
| **180.71** | 77.88 | 102.83 | 5.008% | 0.8 | Livre de Risco ($F e^{-rT}$) |

---

## 5. Como Executar

### 5.1. Pré-requisitos

Instale os pacotes necessários:

```bash
pip install torch numpy matplotlib
```

### 5.2. Execução da Simulação

Execute o código completo de treinamento da PINN e extração de spreads:

```bash
python3 black_cox_pinn_credit.py
```

---

## 6. Referências Bibliográficas

1. **Wilcke, L. T. (2026).** *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel*. Capítulo 36: Aprendizado de Máquina Físico-Informado (PINNs) em Equações Diferenciais[cite: 2].
2. **Black, F., & Cox, J. C. (1976).** *Valuing corporate securities: Some effects of bond indenture provisions*. The Journal of Finance, 31(2), 351-367.
3. **Merton, R. C. (1974).** *On the pricing of corporate debt: The risk structure of interest rates*. The Journal of Finance, 29(2), 449-470.
4. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707.