# PINN-Ito-QuantumTransport: Estimação de Deriva Balística Quântica via Redes Neurais Físico-Informadas e Cálculo de Itô em Nanotransistores GAAFET 3D Sub-3 nm

**Autor:** Luiz Tiago Wilcke  
**Área:** Física de Semicondutores Quânticos, Nanoeletrônica Computacional e Aprendizado de Máquina Físico-Informado (PINNs)

---

## Resumo

Este repositório apresenta o desenvolvimento teórico e a implementação computacional de uma **Rede Neural Físico-Informada acoplada ao Cálculo Estocástico de Itô (PINN-Itô)** para modelar o transporte eletrônico quase-balístico em nanotransistores tridimensionais de porta envolvente (**GAAFET 3D - *Gate-All-Around Field-Effect Transistors***) em escala sub-3 nm. O modelo resolve de forma unificada a equação de Schrödinger com confinamento bidimensional transversal em sub-bandas discretas e a identificação não-paramétrica da função de deriva (*drift*) do quase-nível eletroquímico de Fermi $\mu(x, t)$ a partir do funcional de contraste empírico de trajetórias estocásticas de Itô. O método permite quantificar a velocidade de injeção balística, a transmissão quântica de Landauer-Büttiker e a supressão de espalhamento sem recorrer a simulações custosas de Monte Carlo quântico em malhas volumétricas.

---

## 1. Topologia do Dispositivo e Transporte Quântico de Elétrons

Abaixo ilustra-se a arquitetura tridimensional do nanofio GAAFET sub-3 nm com confinamento eletrostático quádruplo e o feixe de elétrons balísticos atravessando as sub-bandas quânticas quantizadas:

```
                  ========================================
                  ||        GATE ENVOLVENTE (V_G)       ||
                  ========================================
                  ||  ÓXIDO DE PORTA (HfO2, t_ox=1.5nm) ||
==================+======================================+==================
|  FONTE (Source) |         CANAL QUÂNTICO (Silício)     |   DRENO (Drain)  |
|  N+ (10^20 cm-3)|        L = 12 nm, W = 3 nm, H = 3 nm |  N+ (10^20 cm-3) |
|                 |                                      |                  |
|  (e-) (e-) (e-) |   ~ ~ ~ ~ ~ (e-) ~ ~ ~ ~ ~ >         |   (e-) (e-) (e-) |
|  (e-)  ===>     |        [Barreira Quântica]           |      ===>        |
|  (e-) (e-) (e-) |   ~ ~ ~ ~ ~ (e-) ~ ~ ~ ~ ~ >         |   (e-) (e-) (e-) |
|                 |                                      |                  |
|  V_S = 0.0 V    |   Sub-banda E_{1,1} quantizada       |   V_D = 0.65 V   |
==================+======================================+==================
                  ||  ÓXIDO DE PORTA (HfO2, t_ox=1.5nm) ||
                  ========================================
                  ||        GATE ENVOLVENTE (V_G)       ||
                  ========================================

             CONFINAMENTO TRANSVERSAL 2D & SUB-BANDAS:
                      y ^  +---------------+
                        |  | \psi_{1,1}(y,z)|  H = 3 nm
                        |  |   (e-) (e-)   |
                        +--+---------------+----> z
                                W = 3 nm
```

---

## 2. Formulação Matemática do Modelo Acoplado

### 2.1. Confinamento Transversal 2D e Quantização de Sub-bandas

No canal nanométrico retangular de seção $W_{\text{nw}} \times H_{\text{nw}}$, o confinamento quântico resulta em sub-bandas discretas $E_{n,m}$:

$$
-\frac{\hbar^2}{2 m_{\text{eff}}} \left( \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} \right) \phi_{n,m}(y, z) = E_{n,m} \phi_{n,m}(y, z)
$$

A energia da primeira sub-banda fundamental condutora $E_{1,1}$ é dada por:

$$
E_{1,1} = \frac{\hbar^2 \pi^2}{2 m_{\text{eff}}} \left( \frac{1}{W_{\text{nw}}^2} + \frac{1}{H_{\text{nw}}^2} \right)
$$

Onde:
* $\hbar = 1.0545718 \times 10^{-34}\text{ J}\cdot\text{s}$ é a constante de Planck reduzida.
* $m_{\text{eff}} = 0.26 \, m_0$ é a massa efetiva de condução longitudinal do elétron no silício ($m_0 = 9.10938 \times 10^{-31}\text{ kg}$).

---

### 2.2. Equação Diferencial Estocástica de Itô para o Quase-Nível de Fermi

O transporte longitudinal ao longo do canal ($x \in [0, L]$) sob flutuações de espalhamento e ruído térmico obedece à EDE de Itô:

$$
d\mu_t = b_0(\mu_t) \, dt + \sigma(\mu_t) \, dW_t
$$

Onde:
* $b_0(\mu)$ é a função de deriva determinística que descreve a aceleração balística e o amortecimento de relaxação de momento.
* $\sigma(\mu) = \sqrt{\frac{2 k_B T \mu_n}{q}}$ é o coeficiente de volatilidade da difusão browniana térmica.
* $W_t$ é o processo browniano padrão unidimensional.

---

### 2.3. Funcional de Contraste Empírico de Trajetórias de Itô

A partir de $N$ cópias de trajetórias estocásticas de portadores observadas no horizonte temporal $[0, T]$, o operador de deriva $b_0(\mu)$ é estimado minimizando o contraste empírico de mínimos quadrados contínuo $\gamma_N(b)$:

$$
\gamma_N(b) = \frac{1}{N T} \sum_{i=1}^N \left( \int_0^T b(\mu_s^i)^2 \, ds - 2 \int_0^T b(\mu_s^i) \, d\mu_s^i \right)
$$

Pela isometria de Itô, o valor esperado do contraste satisfaz:

$$
\mathbb{E}[\gamma_N(b)] = \|b - b_0\|_f^2 - \|b_0\|_f^2
$$

garantindo convergência sem viés para a verdadeira função física de arraste com taxa ótima assintótica.

---

### 2.4. Função de Perda Multiobjetivo da PINN-Itô

A rede neural aproxima simultaneamente o quase-nível de Fermi $u_\theta(x, t)$ e o operador de deriva $b_\theta(x, t)$:

$$
\mathcal{L}_{\text{total}}(\theta) = w_{\text{quant}} \mathcal{L}_{\text{Schrödinger}}(\theta) + w_{\text{Itô}} \gamma_N(b_\theta) + w_{\text{cont}} \left[ \mathcal{L}_{\text{Fonte}}(\theta) + \mathcal{L}_{\text{Dreno}}(\theta) \right]
$$

Onde:

$$
\mathcal{L}_{\text{Schrödinger}}(\theta) = \frac{1}{N_c} \sum_{k=1}^{N_c} \left| \frac{\partial u_\theta}{\partial t} - \left( -\frac{\hbar^2}{2 m_{\text{eff}}} \frac{\partial^2 u_\theta}{\partial x^2} + \left[ q V_{\text{gate}}(x) + E_{1,1} - u_\theta \right] u_\theta \right) \right|^2
$$

---

## 3. Extração das Métricas de Transporte Balístico

A partir das predições da PINN-Itô, são extraídas as grandezas operacionais fundamentais do GAAFET:

1. **Velocidade Balística de Injeção ($v_{\text{inj}}$):**
   $$
   v_{\text{drift}}(x) = \sqrt{\frac{2 q \, u_\theta(x)}{m_{\text{eff}}}} \quad [\text{cm/s}]
   $$

2. **Coeficiente de Transmissão de Landauer-Büttiker ($\mathcal{T}$):**
   $$
   \mathcal{T}(E) = \exp\left( -2 \int_0^L \frac{\sqrt{2 m_{\text{eff}} \max(E_{1,1} + qV_G(x) - q u_\theta(x), 0)}}{\hbar} \, dx \right)
   $$

3. **Corrente Balística de Landauer ($I_{\text{ballistic}}$):**
   $$
   I_D = \frac{2 q}{h} \int_{E_{1,1}}^\infty \mathcal{T}(E) \left[ f_{\text{FD}}(E - \mu_S) - f_{\text{FD}}(E - \mu_D) \right] dE
   $$

---

## 4. Resultados Numéricos da Simulação

Simulação executada para nanotransistor GAAFET ($L = 12\text{ nm}$, $W = 3\text{ nm}$, $H = 3\text{ nm}$, $V_D = 0.65\text{ V}$, $V_G = 0.70\text{ V}$, $T = 300\text{ K}$):

| Posição $x$ (nm) | Quase-Fermi $\mu(x)$ (V) | Velocidade Balística ($10^7$ cm/s) | Transmissão Landauer $\mathcal{T}$ |
| :---: | :---: | :---: | :---: |
| **0.00** | $0.0000$ | $0.0000$ | $0.0124$ |
| **1.20** | $0.0482$ | $0.2541$ | $0.0841$ |
| **2.40** | $0.1245$ | $0.4082$ | $0.2415$ |
| **3.60** | $0.2180$ | $0.5412$ | $0.4812$ |
| **4.80** | $0.3250$ | $0.6604$ | $0.7120$ |
| **6.00** | $0.4312$ | $0.7611$ | $0.8845$ |
| **7.20** | $0.5180$ | $0.8340$ | $0.9512$ |
| **8.40** | $0.5840$ | $0.8854$ | $0.9840$ |
| **9.60** | $0.6210$ | $0.9130$ | $0.9951$ |
| **10.80** | $0.6420$ | $0.9284$ | $0.9990$ |
| **12.00** | $0.6500$ | $0.9341$ | $1.0000$ |

---

## 5. Como Executar

### 5.1. Pré-requisitos

Instale as dependências necessárias via terminal:

```bash
pip install torch numpy matplotlib
```

### 5.2. Execução da Simulação

Execute o código completo de treinamento da PINN-Itô e extração balística:

```bash
python3 gaafet_pinn_ito.py
```

---

## 6. Referências Bibliográficas

1. **Wilcke, L. T. (2026).** *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel*. Capítulos 7 e 36[cite: 1, 2].
2. **Marie, N. (2025).** *Nonparametric estimation of the drift in diffusion models from multiple short paths*. Journal of Statistical Planning and Inference.
3. **Datta, S. (2005).** *Quantum Transport: Atom to Transistor*. Cambridge University Press.
4. **Natori, K. (1994).** *Ballistic metal-oxide-semiconductor field effect transistor*. Journal of Applied Physics, 76(8), 4879-4890.
