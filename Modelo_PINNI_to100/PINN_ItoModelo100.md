# PINN-Ito-QuantumTransport: Simulação de Transporte Quase-Balístico via Redes Neurais Físico-Informadas e Cálculo de Itô em Nanotransistores GAAFET Sub-2 nm

**Autor:** Luiz Tiago Wilcke  
**Área:** Física de Semicondutores Quânticos, Nanoeletrônica Computacional e Redes Neurais Físico-Informadas (PINNs)

---

## Resumo

Este repositório apresenta o desenvolvimento teórico e a implementação computacional de uma **Rede Neural Físico-Informada acoplada ao Cálculo Estocástico de Itô (PINN-Itô)** para modelar o transporte eletrônico quase-balístico em transistores de efeito de campo de porta envolvente (**GAAFET 3D - *Gate-All-Around Field-Effect Transistors***) em nós tecnológicos sub-2 nm (arquiteturas aplicadas em GPUs de alta densidade como NVIDIA Blackwell/Rubin). O modelo resolve de forma simultânea a equação de Schrödinger com confinamento bidimensional transversal em sub-bandas discretas e a identificação não-paramétrica da função de deriva (*drift*) do quase-nível eletroquímico de Fermi $\mu(x, t)$ a partir do funcional de contraste empírico de trajetórias estocásticas de Itô. A metodologia permite a extração instantânea e sem malha da velocidade de injeção balística, densidade quântica de carga e transmissão de Landauer-Büttiker.

---

## 1. Topologia do Dispositivo e Transporte Eletrônico Quântico

O dispositivo simulado corresponde a um nanofio semicondutor de Silício envolto concentricamente por uma camada de óxido de alta constante dielétrica ($\text{HfO}_2$, $\text{EOT} = 0.6\text{ nm}$) e porta metálica em todas as quatro faces:

```
+=============================================================================+
|                      GATE ENVOLVENTE 3D (V_G = 0.70 V)                      |
|                 +-----------------------------------------+                 |
|                 |   ÓXIDO HIGH-k (HfO2, EOT = 0.6 nm)     |                 |
+=================+=========================================+=================+
|  FONTE (Source) |          CANAL QUÂNTICO (Si 2 nm)       |  DRENO (Drain)  |
|  N+ = 10^20 cm-3|        L = 10 nm, W = 2.5 nm, H = 2.5 nm|  N+ = 10^20 cm-3|
|                 |                                         |                 |
|  (e-) (e-) (e-) |   ~ ~ ~ ~ ~ (e-) ~ ~ ~ ~ ~ >            |  (e-) (e-) (e-) |
|       ===>      |        [Barreira Quântica]              |       ===>      |
|  (e-) (e-) (e-) |   ~ ~ ~ ~ ~ (e-) ~ ~ ~ ~ ~ >            |  (e-) (e-) (e-) |
|                 |                                         |                 |
|  V_S = 0.00 V   |      Sub-banda E_{1,1} quantizada       |  V_D = 0.65 V   |
+=================+=========================================+=================+
|                 |   ÓXIDO HIGH-k (HfO2, EOT = 0.6 nm)     |                 |
|                 +-----------------------------------------+                 |
|                      GATE ENVOLVENTE 3D (V_G = 0.70 V)                      |
+=============================================================================+

             CONFINAMENTO TRANSVERSAL 2D & SUB-BANDAS:
                      y ^  +---------------+
                        |  | \psi_{1,1}(y,z)|  H = 2.5 nm
                        |  |   (e-) (e-)   |
                        +--+---------------+----> z
                                W = 2.5 nm
```

* **Comprimento do Canal ($L$):** $10.0\text{ nm}$ ($x \in [0, L]$)
* **Largura do Nanofio ($W$):** $2.5\text{ nm}$
* **Altura do Nanofio ($H$):** $2.5\text{ nm}$
* **Espessura Equivalente de Óxido ($\text{EOT}$):** $0.6\text{ nm}$
* **Tensão de Dreno ($V_{DD}$):** $0.65\text{ V}$
* **Tensão de Porta ($V_{GS}$):** $0.70\text{ V}$

---

## 2. Formulação Matemática do Modelo Acoplado

### 2.1. Confinamento Transversal 2D e Quantização de Sub-bandas

A função de onda tridimensional fatora-se na base das autofunções transversais:

$$
\Psi(x, y, z, t) = \sum_{n, m} \psi_{n,m}(x, t) \, \phi_{n,m}(y, z)
$$

A equação de Schrödinger transversal em $\Omega_{\text{trans}} = [0, W] \times [0, H]$ determina a energia de corte da primeira sub-banda condutora $E_{1,1}$:

$$
-\frac{\hbar^2}{2 {m_{\text{eff}}}} \left( \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} \right) \phi_{n,m}(y, z) = E_{n,m} \phi_{n,m}(y, z)
$$

$$
E_{1,1} = \frac{\hbar^2 \pi^2}{2 {m_{\text{eff}}}} \left( \frac{1}{W^2} + \frac{1}{H^2} \right)
$$

Onde:
* $\hbar = 1.0545718 \times 10^{-34}\text{ J}\cdot\text{s}$ é a constante de Planck reduzida.
* ${m_{\text{eff}}} = 0.26 \, m_0$ é a massa efetiva de condução no silício ($m_0 = 9.10938 \times 10^{-31}\text{ kg}$).

---

### 2.2. Equação Quântica de Schrödinger 1D Longitudinal

A função de onda complexa $\psi(x, t) = \psi_R(x, t) + i \psi_I(x, t)$ acopla-se ao potencial auto-consistente $V_{\text{gate}}(x)$ e ao quase-nível de Fermi $\mu(x, t)$:

$$
i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2 {m_{\text{eff}}}} \frac{\partial^2 \psi}{\partial x^2} + \left[ q V_{\text{gate}}(x) + E_{1,1} - q \mu(x, t) \right] \psi
$$

Decompondo nas partes real e imaginária:

$$
\hbar \frac{\partial \psi_R}{\partial t} = -\frac{\hbar^2}{2 {m_{\text{eff}}}} \frac{\partial^2 \psi_I}{\partial x^2} + \left[ q V_{\text{gate}}(x) + E_{1,1} - q \mu(x, t) \right] \psi_I
$$

$$
-\hbar \frac{\partial \psi_I}{\partial t} = -\frac{\hbar^2}{2 {m_{\text{eff}}}} \frac{\partial^2 \psi_R}{\partial x^2} + \left[ q V_{\text{gate}}(x) + E_{1,1} - q \mu(x, t) \right] \psi_R
$$

---

### 2.3. Dinâmica de Itô do Quase-Nível de Fermi

O transporte estocástico sob flutuações térmicas de espalhamento obedece à Equação Diferencial Estocástica de Itô:

$$
d\mu_t = b_0(\mu_t) \, dt + \sigma(\mu_t) \, dW_t
$$

Onde:
* $b_0(\mu)$ é a função de deriva (*drift*) determinística de arraste balístico.
* $\sigma(\mu) = \sqrt{\frac{2 k_B T \mu_n}{q}}$ é o coeficiente de volatilidade térmica de difusão.
* $W_t$ é o processo browniano padrão unidimensional.

---

### 2.4. Contraste Empírico de Trajetórias de Itô ($\gamma_N$)

A partir de $N$ cópias de trajetórias microscópicas observadas no horizonte $[0, T]$, o operador de deriva $b_0(\mu)$ é estimado minimizando o funcional de contraste de mínimos quadrados contínuo de Marie:

$$
\gamma_N(b) = \frac{1}{NT} \sum_{i=1}^N \left( \int_0^T b(\mu_s^i)^2 \, ds - 2 \int_0^T b(\mu_s^i) \, d\mu_s^i \right)
$$

Pela isometria de Itô:

$$
\mathbb{E}[\gamma_N(b)] = \|b - b_0\|_f^2 - \|b_0\|_f^2
$$

---

### 2.5. Função de Perda Multiobjetivo da PINN-Itô

A rede neural aproxima simultaneamente a função de onda complexa, o quase-nível de Fermi e a deriva:

$$
\mathcal{L}_{\text{total}}(\theta) = w_{\text{Sch}} \mathcal{L}_{\text{Schrödinger}}(\theta) + w_{\text{Itô}} \gamma_N(b^\theta) + w_{\text{BC}} \left( \mathcal{L}_{\text{Fonte}}(\theta) + \mathcal{L}_{\text{Dreno}}(\theta) \right)
$$

Onde:

$$
\mathcal{L}_{\text{Schrödinger}}(\theta) = \frac{1}{N_c} \sum_{k=1}^{N_c} \left( \left| \hbar \frac{\partial \psi_R^\theta}{\partial t} - \hat{\mathcal{H}}_I \psi_I^\theta \right|^2 + \left| -\hbar \frac{\partial \psi_I^\theta}{\partial t} - \hat{\mathcal{H}}_R \psi_R^\theta \right|^2 \right)
$$

---

## 3. Extração das Figuras de Mérito Balísticas

1. **Velocidade Balística de Injeção ($v_{\text{drift}}$):**
   $$
   v_{\text{drift}}(x) = \sqrt{\frac{2 q \, \mu^\theta(x)}{{m_{\text{eff}}}}} \quad [\text{cm/s}]
   $$

2. **Coeficiente de Transmissão Quântica ($\mathcal{T}$):**
   $$
   \mathcal{T}(E) = \exp\left( -2 \int_0^L \frac{\sqrt{2 {m_{\text{eff}}} \max\left( E_{1,1} + q V_{\text{gate}}(x) - E, \, 0 \right)}}{\hbar} \, dx \right)
   $$

3. **Corrente de Dreno Balística de Landauer-Büttiker ($I_D$):**
   $$
   I_D = \frac{2 q}{h} \int_{E_{1,1}}^\infty \mathcal{T}(E) \left[ \frac{1}{1 + \exp\left( \frac{E - q\mu_S}{k_B T} \right)} - \frac{1}{1 + \exp\left( \frac{E - q\mu_D}{k_B T} \right)} \right] dE
   $$

---

## 4. Resultados Numéricos da Simulação

Resultados obtidos para o nanotransistor GAAFET 2 nm ($L = 10\text{ nm}$, $W = 2.5\text{ nm}$, $H = 2.5\text{ nm}$, $V_D = 0.65\text{ V}$, $V_G = 0.70\text{ V}$, $T = 300\text{ K}$):

| Posição $x$ (nm) | Quase-Fermi $\mu(x)$ (V) | Velocidade Balística ($10^7$ cm/s) | Transmissão Landauer $\mathcal{T}$ |
| :---: | :---: | :---: | :---: |
| **0.00** | $0.0028$ | $0.6108$ | $0.7019$ |
| **1.00** | $0.4719$ | $7.9900$ | $0.8492$ |
| **2.00** | $0.7342$ | $9.9663$ | $0.9706$ |
| **3.00** | $0.9519$ | $11.3484$ | $1.0000$ |
| **4.00** | $0.9735$ | $11.4765$ | $0.9231$ |
| **5.00** | $0.8659$ | $10.8238$ | $0.7810$ |
| **6.00** | $0.6435$ | $9.3306$ | $0.6940$ |
| **7.00** | $0.3645$ | $7.0220$ | $0.6380$ |
| **8.00** | $0.3432$ | $6.8145$ | $0.6811$ |
| **9.00** | $0.3594$ | $6.9736$ | $0.7787$ |
| **10.00** | $0.6500$ | $9.3410$ | $1.0000$ |

---

## 5. Como Executar

### 5.1. Pré-requisitos

Instale os pacotes necessários via terminal:

```bash
pip install torch numpy matplotlib
```

### 5.2. Execução da Simulação

Execute o script de treinamento e extração física:

```bash
python3 gaafet_2nm_pinn_ito.py
```

---

## 6. Referências Bibliográficas

1. **Wilcke, L. T. (2026).** *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel*. Capítulos 6, 7 e 36.
2. **Marie, N. (2025).** *Nonparametric estimation of the drift in diffusion models from multiple short paths*. Journal of Statistical Planning and Inference.
3. **Datta, S. (2005).** *Quantum Transport: Atom to Transistor*. Cambridge University Press.
4. **Natori, K. (1994).** *Ballistic metal-oxide-semiconductor field effect transistor*. Journal of Applied Physics, 76(8), 4879-4890.