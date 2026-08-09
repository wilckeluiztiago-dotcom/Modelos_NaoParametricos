# Pipeline Computacional: Nanosensor Supercondutor para Neutrinos (CSNND)

**Inferência Estatística Não-Paramétrica em Equações Diferenciais Estocásticas de Itô-Lévy com Saltos**

Este repositório implementa um pipeline completo de simulação e estimação não-paramétrica para o **Nanosensor Supercondutor para Neutrinos (CSNND)**. O código simula trajetórias de recuo nuclear induzidas por interações coerentes elásticas neutrino-núcleo (CEvNS) e estima a função de deriva via projeção ortogonal sobre bases de Legendre.

> **Origem Teórica**  
> Toda a teoria matemática subjacente — Equações Diferenciais Estocásticas com saltos, estimadores de projeção, matrizes de design empíricas, estabilidade espectral, cálculo de Malliavin, U-estatísticas degeneradas e teoria minimax — foi extraída integralmente do livro:  
> **Wilcke, A. (2026).** *Métodos Avançados em Inferência Estatística Não-Paramétrica*.

---

## Sumário

1. [Descrição do Modelo Físico](#1-descrição-do-modelo-físico)
2. [Equação Diferencial Estocástica de Itô-Lévy](#2-equação-diferencial-estocástica-de-itô-lévy)
3. [Discretização de Euler-Maruyama com Saltos](#3-discretização-de-euler-maruyama-com-saltos)
4. [Base Ortonormal de Legendre](#4-base-ortonormal-de-legendre)
5. [Estimador de Projeção de Mínimos Quadrados](#5-estimador-de-projeção-de-mínimos-quadrados)
6. [Estabilidade Espectral e Regularização](#6-estabilidade-espectral-e-regularização)
7. [U-Estatísticas Degeneradas](#7-u-estatísticas-degeneradas)
8. [Cálculo de Malliavin e Isometria de Skorokhod](#8-cálculo-de-malliavin-e-isometria-de-skorokhod)
9. [Risco Integrado $L^2$](#9-risco-integrado-l2)
10. [Parâmetros Físicos do CSNND](#10-parâmetros-físicos-do-csnnd)
11. [Instalação e Execução](#11-instalação-e-execução)
12. [Estrutura do Código](#12-estrutura-do-código)
13. [Referências Teóricas (Wilcke, 2026)](#13-referências-teóricas-wilcke-2026)

---

## 1. Descrição do Modelo Físico

O CSNND opera em regime sub-Kelvin e detecta recuos nucleares coerentes provocados por neutrinos. A amplitude de pulso de quasipartículas $X_t$ é modelada como um processo de difusão com saltos de Poisson composto, onde:

- A **deriva** $b_0(x)$ representa a relaxação eletrotérmica do nanofio supercondutor.
- A **volatilidade** $\sigma_0(x)$ modela o ruído quântico de fundo.
- Os **saltos** $\Delta J_t$ modelam os recuos nucleares discretos induzidos por eventos CEvNS.

---

## 2. Equação Diferencial Estocástica de Itô-Lévy

O processo de estado $X = (X_t)_{t \in [0,T]}$ satisfaz a EDE de Itô-Lévy:

$$
dX_t = b_0(X_t)\,dt + \sigma_0(X_t)\,dW_t + dJ_t,
$$

onde:

- $W = (W_t)_{t \geq 0}$ é um movimento browniano padrão unidimensional;
- $J = (J_t)_{t \geq 0}$ é um processo de Poisson composto (saltos de CEvNS):

$$
J_t = \sum_{i=1}^{N_t} \xi_i,
$$

com $N_t$ um processo de Poisson de intensidade $\lambda > 0$ e $\xi_i \sim \mathcal{N}(0,\varsigma^2)$ i.i.d. (amplitude de recuo nuclear).

### Função de Deriva Teórica

$$
b_0(x) = -3.8\,x + 1.5\,x^3 - 0.5\,\tanh(2x).
$$

### Coeficiente de Difusão Local

$$
\sigma_0(x) = 0.30\bigl(1 + 0.15\cos^2(x)\bigr).
$$

---

## 3. Discretização de Euler-Maruyama com Saltos

No horizonte temporal fixo $[0,T]$ com partição uniforme de $n$ passos ($\Delta t = T/n$), a discretização de Euler-Maruyama estendida é:

$$
X_{k+1} = X_k + b_0(X_k)\,\Delta t + \sigma_0(X_k)\,\Delta W_k + \Delta J_k,
\quad k = 0,\dots,n-1,
$$

onde:

$$
\Delta W_k \sim \mathcal{N}(0,\Delta t),
\qquad
\Delta J_k =
\begin{cases}
\xi_k & \text{com probabilidade }\lambda\Delta t,\\
0 & \text{caso contrário},
\end{cases}
\quad
\xi_k \sim \mathcal{N}(0,0.60^2).
$$

São geradas $N$ trajetórias independentes $(X^{(i)})_{i=1}^N$ (paradigma de múltiplas trajetórias curtas — Capítulo 7.1).

---

## 4. Base Ortonormal de Legendre

A estimação é realizada no subespaço de dimensão finita $S_m = \mathrm{span}\{\varphi_0,\dots,\varphi_{m-1}\}$, onde $\{\varphi_j\}$ é o sistema ortonormal de Legendre (após mudança de escala $x\mapsto x/1.5$ para o intervalo $[-1,1]$):

$$
\begin{align*}
\varphi_0(x) &= \sqrt{\tfrac{1}{2}},\\
\varphi_1(x) &= \sqrt{\tfrac{3}{2}}\,x,\\
\varphi_2(x) &= \sqrt{\tfrac{5}{2}}\cdot\tfrac{1}{2}(3x^2-1),\\
\varphi_3(x) &= \sqrt{\tfrac{7}{2}}\cdot\tfrac{1}{2}(5x^3-3x),\\
\varphi_4(x) &= \sqrt{\tfrac{9}{2}}\cdot\tfrac{1}{8}(35x^4-30x^2+3),\\
\varphi_5(x) &= \sqrt{\tfrac{11}{2}}\cdot\tfrac{1}{8}(63x^5-70x^3+15x).
\end{align*}
$$

Em geral,

$$
\varphi_j(x) = \sqrt{j+\tfrac{1}{2}}\,P_j(x),
$$

onde $P_j$ são os polinômios de Legendre clássicos. A dimensão utilizada no código é $m=6$.

---

## 5. Estimador de Projeção de Mínimos Quadrados

### Função Objetivo Estocástica

O estimador minimiza o risco empírico contínuo (Capítulo 3, Seção 3.4.1):

$$
\gamma_N(b) = \frac{1}{NT}\sum_{i=1}^N\int_0^T \bigl|b(X_s^{(i)})\bigr|^2\,ds
- \frac{2}{NT}\sum_{i=1}^N\int_0^T b(X_s^{(i)})\,dX_s^{(i)}.
$$

### Matriz de Design Empírica Contínua

$$
\hat{\Psi}_m = \frac{1}{NT}\sum_{i=1}^N\sum_{k=0}^{n-1}
\varphi(X_k^{(i)})\varphi(X_k^{(i)})^\top \Delta t
\quad \in \mathbb{R}^{m\times m}
\qquad\text{(Equação 7.5)}.
$$

### Vetor de Dados Estocástico

$$
\hat{Z}_m = \frac{1}{NT}\sum_{i=1}^N\sum_{k=0}^{n-1}
\varphi(X_k^{(i)})\,\Delta X_k^{(i)}
\quad \in \mathbb{R}^m
\qquad\text{(Equação 7.6)},
$$

onde $\Delta X_k^{(i)} = X_{k+1}^{(i)} - X_k^{(i)}$.

### Estimador de Projeção

O estimador de coeficientes é a solução do sistema linear:

$$
\hat{\theta}_m = \hat{\Psi}_m^{-1}\hat{Z}_m,
$$

e a função de deriva estimada é:

$$
\hat{b}_m(x) = \hat{\theta}_m^\top \varphi(x) = \sum_{j=0}^{m-1}\hat{\theta}_{m,j}\,\varphi_j(x).
$$

---

## 6. Estabilidade Espectral e Regularização

A estabilidade da matriz empírica é monitorada pelos autovalores:

$$
\lambda_{\min}(\hat{\Psi}_m)\qquad\text{e}\qquad\lambda_{\max}(\hat{\Psi}_m).
$$

Para garantir invertibilidade (Desigualdade de Bernstein Matricial — Capítulo 24.6, Teorema 24.2), aplica-se regularização ridge:

$$
\hat{\Psi}_m^{\mathrm{reg}} = \hat{\Psi}_m + \varepsilon\cdot\lambda_{\max}(\hat{\Psi}_m)\,I_m,
\qquad\varepsilon = 10^{-5}.
$$

O estimador regularizado torna-se:

$$
\hat{\theta}_m^{\mathrm{reg}} = \bigl(\hat{\Psi}_m^{\mathrm{reg}}\bigr)^{-1}\hat{Z}_m.
$$

---

## 7. U-Estatísticas Degeneradas

Para análise de flutuações de segunda ordem (Capítulo 17), considera-se a U-estatística degenerada contínua de ordem 2:

$$
U_N(H_2) = \binom{N}{2}^{-1}\sum_{1\leq i < j\leq N} H_2\bigl(X^{(i)},X^{(j)}\bigr),
$$

com núcleo simétrico baseado na decomposição de Hoeffding:

$$
H_2\bigl(X^{(i)},X^{(j)}\bigr)
= \Biggl(\frac{1}{T}\int_0^T\bigl(X_t^{(i)}-X_t^{(j)}\bigr)^2\,dt\Biggr)
\exp\Biggl(-0.2\cdot\mathrm{Var}\bigl(X^{(i)}-X^{(j)}\bigr)\Biggr).
$$

O Teorema do Limite Central para U-estatísticas degeneradas (Capítulo 17, Seção 17.3) justifica a normalidade assintótica das flutuações.

---

## 8. Cálculo de Malliavin e Isometria de Skorokhod

Para processos não-adaptados, a isometria de Skorokhod (Capítulo 25, Equação 25.8) fornece a correção de variância:

$$
\mathbb{E}\bigl[\delta(u)^2\bigr]
= \mathbb{E}\bigl[\|u\|_{L^2([0,T])}^2\bigr]
+ \mathbb{E}\bigl[\langle Du,Du\rangle_{L^2([0,T]^2)}\bigr],
$$

onde $\delta$ é a integral de Skorokhod e $D$ a derivada de Malliavin. No código, a correção numérica é implementada como:

$$
\mathrm{Var}_{\mathrm{Skorokhod}}
= \mathrm{Var}_{\mathrm{base}} + 0.15\cdot\mathrm{Var}_{\mathrm{base}}.
$$

---

## 9. Risco Integrado $L^2$

O risco de estimação é avaliado pela norma $L^2$ integrada sobre uma grade espacial $\mathcal{G}$:

$$
R_N(\hat{b}_m)
= \int_{\mathcal{G}}\bigl(b_0(x)-\hat{b}_m(x)\bigr)^2\,dx
\approx \mathrm{trapezoid}\bigl((b_0-\hat{b}_m)^2,\mathcal{G}\bigr).
$$

(Referência: Capítulo 3, Seção 3.4.4 e Capítulo 7, Seção 7.5.)

---

## 10. Parâmetros Físicos do CSNND

| Parâmetro                    | Símbolo / Valor                  | Unidade | Descrição                                      |
|-----------------------------|----------------------------------|---------|------------------------------------------------|
| Temperatura do banho        | $T_{\mathrm{bath}}=0.05$         | K       | Operação sub-Kelvin                            |
| Gap supercondutor           | $\Delta=1.8\times 10^{-22}$      | J       | Gap de energia do nanofio                      |
| Massa nuclear efetiva       | $m_N=1.5\times 10^{-25}$         | kg      | Massa efetiva para CEvNS                       |
| Seção de choque coerente    | $\sigma_{\nu N}=4.2\times 10^{-44}$ | cm²  | Seção de choque neutrino-núcleo                |
| Constante de Boltzmann      | $k_B=1.380649\times 10^{-23}$    | J/K     | —                                              |
| Intensidade do fluxo        | $\lambda=0.45$                   | s⁻¹     | Taxa média de colisões CEvNS                   |
| Desvio-padrão do recuo      | $\varsigma=0.60$                 | —       | Amplitude típica do salto                      |

---

## 11. Instalação e Execução

### Requisitos

```bash
pip install numpy scipy matplotlib
```

### Execução

```bash
python csnnd_pipeline.py
```

O pipeline:

1. Simula $N=160$ trajetórias de comprimento $n=1500$ passos;
2. Constrói a base de Legendre de dimensão $m=6$;
3. Estima $\hat{b}_m$ via projeção de mínimos quadrados regularizada;
4. Avalia U-estatísticas degeneradas e a isometria de Skorokhod;
5. Calcula o risco $L^2$ e gera o gráfico comparativo $b_0(x)$ vs. $\hat{b}_m(x)$.

---

## 12. Estrutura do Código

```
csnnd-neutrinos-pipeline/
├── csnnd_pipeline.py          # Pipeline completo
└── README.md                  # Este arquivo
```

### Classes Principais

| Classe                                      | Responsabilidade                                      |
|---------------------------------------------|-------------------------------------------------------|
| `ParametrosFisicosCSNND`                    | Constantes físicas e criogênicas                      |
| `SimuladorEstocasticoCSNNDLevy`             | Simulação de EDEs de Itô-Lévy com saltos              |
| `BaseOrthonormalLegendreCSNND`              | Sistema ortonormal de Legendre                        |
| `EstimadorProjecaoMinimosQuadradosNeutrinos`| Estimador de projeção + regularização espectral       |
| `AnalisadorUStatisticsDegeneradasCSNND`     | U-estatísticas degeneradas de 2ª ordem                |
| `SimuladorMalliavinSkorokhodCSNND`          | Isometria de Skorokhod / correção de Malliavin        |
| `ValidadorRiscoIntegradoCSNND`              | Cálculo do risco $L^2$ integrado                      |

---

## 13. Referências Teóricas (Wilcke, 2026)

| Capítulo / Seção     | Conteúdo Utilizado                                      |
|----------------------|---------------------------------------------------------|
| Cap. 3, §3.4.1       | Função objetivo estocástica $\gamma_N(b)$               |
| Cap. 3, §3.4.4       | Risco integrado $L^2$                                   |
| Cap. 3, §3.5         | Polinômios ortogonais de Jacobi/Legendre                |
| Cap. 5, §5.1–5.5     | Bases ortonormais e estabilidade espectral              |
| Cap. 6, §6.2–6.3     | Fundamentação e simulação Euler-Maruyama                |
| Cap. 7, §7.1         | Paradigma de múltiplas trajetórias curtas independentes |
| Cap. 7, §7.3         | Matriz de design contínua $\hat{\Psi}_m$ e vetor $\hat{Z}_m$ |
| Cap. 7, §7.5         | Validação de risco                                      |
| Cap. 8, §8.1         | Dinâmicas com saltos (Poisson composto)                 |
| Cap. 8, §8.1.1       | Estimação da deriva na presença de saltos               |
| Cap. 17, §17.2–17.3  | Decomposição de Hoeffding e TLC para U-estatísticas     |
| Cap. 24, §24.6       | Desigualdade de Bernstein matricial (Teorema 24.2)      |
| Cap. 25, §25.2–25.4  | Derivada de Malliavin e isometria de Skorokhod          |

---

## Licença e Citação

Se utilizar este código ou as metodologias implementadas, cite:

```
Wilcke, A. (2026). Métodos Avançados em Inferência Estatística Não-Paramétrica.
Pipeline CSNND — Nanosensor Supercondutor para Neutrinos.
```

---

*Repositório preparado para GitHub com todas as equações do modelo renderizadas em LaTeX (macros compatíveis com o MathJax do GitHub).*
