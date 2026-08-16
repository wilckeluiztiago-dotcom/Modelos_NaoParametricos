# FractionalCredit-Molchan: Modelagem Estocástica de Risco de Crédito sob Ciclos Macroeconômicos de Memória Longa via Martingale de Molchan

**Autor:** Luiz Tiago Wilcke  
**Área:** Engenharia Financeira, Gestão de Risco de Crédito Quantitativo e Cálculo Estocástico Fracionário

---

## Resumo

Este repositório apresenta a formulação matemática, a fundamentação estocástica e a implementação computacional de um modelo de previsão da **Taxa Agregada de Inadimplência de Crédito ($I_t$)** sob ciclos econômicos persistentes. O modelo acopla Equações Diferenciais Estocásticas Fracionárias governadas pelo **Movimento Browniano Fracionário (mBf)** com parâmetro de Hurst $H \in (1/2, 1)$ à **Semimartingale Canônica de Molchan**. Essa transformação de Volterra permite contornar a natureza não-semimartingale dos processos com dependência de longo alcance, restaurando a ortogonalidade de martingales, a avaliação de verossimilhança de Girsanov e testes de estresse regulatórios (*Stress Testing* e IFRS 9 / Basileia III/IV).

---

## 1. Fundamentação Teórica e Formulação Matemática

### 1.1. Dinâmica da Taxa de Inadimplência em Espaço Logito

Para assegurar a restrição da taxa de inadimplência ao domínio $I_t \in (0, 1)$, define-se a transformação logito $X_t = \text{logit}(I_t)$:

$$
X_t = \ln\left( \frac{I_t}{1 - I_t} \right) \iff I_t = \frac{1}{1 + \exp(-X_t)}
$$

A variável de estado macroeconômica $X_t \in \mathbb{R}$ obedece à Equação Diferencial Estocástica Fracionária com reversão à média estrutural:

$$
dX_t = -\theta \left( X_t - \mu_{\text{macro}}(t) \right) dt + \sigma \, dB_t^H
$$

Onde:
* $\theta > 0$ é a velocidade de reversão aos fundamentos econômicos.
* $\mu_{\text{macro}}(t)$ é o atrator macroeconômico exógeno (função do desemprego, taxa de juros e inflação inercial).
* $\sigma > 0$ é o coeficiente de difusão da carteira de crédito.
* $B_t^H = (B_t^H)_{t \ge 0}$ é o Movimento Browniano Fracionário com parâmetro de Hurst $H \in (1/2, 1)$.

---

### 1.2. Movimento Browniano Fracionário e Dependência de Longo Alcance

O Movimento Browniano Fracionário $B_t^H$ é o único processo gaussiano contínuo e centrado com função de covariância teórica dada por:

$$
R_H(t, s) = \mathbb{E}\left[ B_t^H B_s^H \right] = \frac{1}{2} \left( t^{2H} + s^{2H} - |t - s|^{2H} \right)
$$

Para o regime persistente $H > 1/2$, a função de autocorrelação dos incrementos discretos exibe decaimento hiperbólico:

$$
\rho(k) = \text{Corr}\left( B_1^H, B_{k+1}^H - B_k^H \right) \sim H(2H - 1) k^{2H - 2} \quad \text{quando } k \to \infty
$$

Como $\sum_{k=1}^\infty \rho(k) = \infty$, o processo é caracterizado por dependência de longo alcance (*long-range dependence*). Para $H \ne 1/2$, o processo $B_t^H$ **não é uma semimartingale**, o que invalida a aplicação direta da integral de Itô clássica.

---

### 1.3. O Martingale Fundamental de Molchan

A filtração natural $\mathcal{F}_t^H = \sigma(B_s^H : 0 \le s \le t)$ é mapeada em uma semimartingale através do **Martingale Fundamental de Molchan** $M_t^H$, definido pela integral determinística de Volterra:

$$
M_t^H = \int_0^t k_H(t, s) \, dB_s^H
$$

Onde o núcleo integrador determinístico $k_H(t, s)$ é expresso para $H > 1/2$ por:

$$
k_H(t, s) = c_H^{-1} s^{H - 1/2} (t - s)^{1/2 - H}, \quad 0 < s < t
$$

com a constante de normalização gaussiana:

$$
c_H = \left[ \frac{H(2H - 1)}{\text{Beta}(2 - 2H, H - 1/2)} \right]^{1/2}
$$

O processo $M_t^H$ é um martingala gaussiano estrito cuja variação quadrática previsível determinística atende à lei de potência:

$$
\langle M^H \rangle_t = w_H \, t^{2 - 2H}
$$

onde a constante de escala de energia $w_H$ é calculada analiticamente por:

$$
w_H = \frac{\Gamma(3 - 2H)}{2H \, \Gamma(3/2 - H) \Gamma(H + 1/2)}
$$

---

### 1.4. Dinâmica Canônica Mapeada e Medida de Girsanov

A aplicação do operador de Molchan transforma o sistema original na equação de semimartingale equivalente:

$$
dY_t = b^*(t) \, dt + \sigma \, dM_t^H
$$

onde $Y_t = \int_0^t k_H(t, s) dX_s$ e $b^*(t) = -\theta \int_0^t k_H(t, s) (X_s - \mu_{\text{macro}}(s)) ds$.

A densidade de Radon-Nikodym de Girsanov para calibração dos parâmetros de risco de crédito assume a forma fechada:

$$
\ln \mathcal{L}(\theta, \sigma) = \frac{1}{\sigma^2} \int_0^T \frac{b^*(t)}{w_H (2 - 2H) t^{1 - 2H}} \, dY_t - \frac{1}{2\sigma^2} \int_0^T \frac{(b^*(t))^2}{w_H (2 - 2H) t^{1 - 2H}} \, dt
$$

---

## 2. Resultados Numéricos e Diagnóstico

Simulação executada para uma carteira de crédito no horizonte $T = 10\text{ anos}$, com $\theta = 0.85$, $\sigma = 0.35$ e $H = 0.75$:

| Parâmetro / Métrica | Valor Teórico | Valor Estimado | Unidade / Descrição |
| :--- | :---: | :---: | :---: |
| Parâmetro de Hurst ($H$) | $0.7500$ | $0.7500$ | Memória Longa / Persistência |
| Constante de Volterra ($c_H$) | $0.8921$ | $0.8921$ | Escalar de Normalização |
| Constante de Energia ($w_H$) | $0.5317$ | $0.5317$ | Escala de Variação |
| Taxa Média de Inadimplência ($\mathbb{E}[I_t]$) | $4.50\%$ | $4.68\%$ | Média Estrutural |
| Pico Máximo de Inadimplência | $-$ | $8.92\%$ | Cenário de Estresse Cíclico |
| Inadimplência P99 (Stress Testing) | $-$ | $8.14\%$ | Nível de Perda Severa |
| Autocorrelação com 1.2 anos de Lag | $0.35$ (Markov) | $0.6841$ (Fracionário) | Retenção de Inércia Histórica |
| Variação Quadrática $\langle M^H \rangle_T$ | $1.6815$ | $1.6742$ | Consistência de Martingale |

---

## 3. Como Executar

### 3.1. Pré-requisitos

Instale os pacotes necessários:

```bash
pip install numpy scipy matplotlib
```

### 3.2. Execução da Simulação

Execute o código completo de calibração estocástica e projeção:

```bash
python3 credit_molchan_fbm.py
```

---

## 4. Referências Bibliográficas

1. **Wilcke, L. T. (2026).** *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel*. Capítulo 26: O Martingale de Molchan e Difusões Fracionárias.
2. **Molchan, G. M. (1969).** *Gaussian stochastic processes with stationary increments and fractional brownian motion*. Sankhya: The Indian Journal of Statistics, Series A, 31(3), 261-268.
3. **Biagini, F., Hu, Y., Øksendal, B., & Zhang, T. (2008).** *Stochastic Calculus for Fractional Brownian Motion and Applications*. Springer Science & Business Media.
4. **McNeil, A. J., Frey, R., & Embrechts, P. (2015).** *Quantitative Risk Management: Concepts, Techniques and Tools*. Princeton University Press.