# Pipeline Computacional: Nanodispositivo Quântico de Alta Frequência

**Inferência Estatística Não-Paramétrica em Equações Diferenciais Estocásticas de Itô**

Este repositório implementa um pipeline completo de simulação e estimação não-paramétrica para um **Nanodispositivo Quântico de Alta Frequência**. O código simula trajetórias do potencial eletrostático / carga no canal e estima a função de deriva via projeção ortogonal sobre bases de Legendre.

> **Origem Teórica**  
> Toda a teoria matemática subjacente — Equações Diferenciais Estocásticas, estimadores de projeção ortogonal, matrizes de design empíricas contínuas, estabilidade espectral e U-estatísticas degeneradas — foi extraída integralmente do livro:  
> **Wilcke, A. (2026).** *Métodos Avançados em Inferência Estatística Não-Paramétrica*.

---

## Sumário

1. [Descrição do Modelo Físico](#1-descrição-do-modelo-físico)
2. [Equação Diferencial Estocástica de Itô](#2-equação-diferencial-estocástica-de-itô)
3. [Paradigma de Múltiplas Trajetórias Curtas](#3-paradigma-de-múltiplas-trajetórias-curtas)
4. [Função Objetivo Quadrática Contínua](#4-função-objetivo-quadrática-contínua)
5. [Matriz de Design e Vetor de Dados](#5-matriz-de-design-e-vetor-de-dados)
6. [Estimador de Projeção Ortogonal](#6-estimador-de-projeção-ortogonal)
7. [Estabilidade Espectral (Bernstein Matricial)](#7-estabilidade-espectral-bernstein-matricial)
8. [U-Estatísticas Degeneradas](#8-u-estatísticas-degeneradas)
9. [Risco Integrado \( L^2 \)](#9-risco-integrado-l2)
10. [Parâmetros Físicos do Nanodispositivo](#10-parâmetros-físicos-do-nanodispositivo)
11. [Instalação e Execução](#11-instalação-e-execução)
12. [Estrutura do Código](#12-estrutura-do-código)
13. [Referências Teóricas (Wilcke, 2026)](#13-referências-teóricas-wilcke-2026)

---

## 1. Descrição do Modelo Físico

O nanodispositivo opera em regime criogênico com confinamento quântico e transporte de alta frequência. A amplitude do potencial efetivo (ou carga) no canal, denotada por \( X_t \), é modelada como um processo de difusão de Itô, onde:

- A **deriva** \( b_0(x) \) representa o potencial eletrostático não-linear e o confinamento quântico.
- A **volatilidade** \( \sigma_0(x) \) modela o ruído de disparo e flutuações térmicas associadas à frequência de plasma dos portadores.

---

## 2. Equação Diferencial Estocástica de Itô

A trajetória do potencial efetivo satisfaz a EDE de Itô unidimensional (Capítulo 6, Seção 6.2, Equação 6.4):

$$
dX_t = b_0(X_t)\, dt + \sigma_0(X_t)\, dW_t, \qquad X_0 = x_0, \qquad t \in [0, T]
$$

onde:

- \( X_t \): potencial eletrostático efetivo / carga no nanodispositivo;
- \( b_0(X_t) \): função de deriva (desconhecida);
- \( \sigma_0(X_t) \): coeficiente de difusão local;
- \( W_t \): movimento browniano padrão.

### Função de Deriva Teórica

$$
b_0(x) = -4.5\, x + 1.8\, x^3 - 0.7\, \tanh(2.0\, x)
$$

### Coeficiente de Difusão Local

$$
\sigma_0(x) = 0.32 \bigl( 1 + 0.18 \cos^2(x) \bigr)
$$

### Discretização de Euler-Maruyama (Equação 6.9)

$$
X_{k+1} = X_k + b_0(X_k)\, \Delta t + \sigma_0(X_k)\, \Delta W_k,
\qquad
\Delta W_k \sim \mathcal{N}(0, \Delta t)
$$

---

## 3. Paradigma de Múltiplas Trajetórias Curtas

Para estimar \( b_0(x) \) sem postular ergodicidade a longo prazo (\( T \to \infty \)), observam-se \( N \) cópias independentes de trajetórias em horizonte temporal fixo \( [0, T] \) (Capítulo 7, Seção 7.1):

$$
dX_t^{(i)} = b_0(X_t^{(i)})\, dt + \sigma_0(X_t^{(i)})\, dW_t^{(i)},
\qquad
X_0^{(i)} = x_0,
\qquad
i = 1, \dots, N
$$

---

## 4. Função Objetivo Quadrática Contínua

O contraste empírico de mínimos quadrados contínuo (Capítulo 3, Seção 3.4.1 e Capítulo 7, Seção 7.2) é:

$$
\gamma_N(b) = \frac{1}{NT} \sum_{i=1}^{N} \left(
\int_0^T b(X_s^{(i)})^2\, ds - 2 \int_0^T b(X_s^{(i)})\, dX_s^{(i)}
\right)
$$

Minimizar \( \gamma_N(b) \) equivale a aproximar a verdadeira deriva eletrostática \( b_0(x) \) do nanodispositivo.

---

## 5. Matriz de Design e Vetor de Dados

Projetando a deriva sobre o subespaço \( \mathcal{S}_m = \mathrm{span}\{\varphi_0, \dots, \varphi_{m-1}\} \) com base ortonormal de Legendre, obtêm-se (Capítulo 7, Seção 7.3, Equações 7.5 e 7.6):

### Matriz de Design Empírica Contínua

$$
\hat{\Psi}_m = \frac{1}{NT} \sum_{i=1}^{N} \int_0^T \varphi(X_s^{(i)}) \varphi(X_s^{(i)})^\top\, ds
\quad \in \mathbb{R}^{m \times m}
$$

### Vetor de Dados Estocástico

$$
\hat{Z}_m = \frac{1}{NT} \sum_{i=1}^{N} \int_0^T \varphi(X_s^{(i)})\, dX_s^{(i)}
\quad \in \mathbb{R}^m
$$

### Base Ortonormal de Legendre (\( m = 6 \))

$$
\begin{align*}
\varphi_0(x) &= \sqrt{\tfrac{1}{2}}, \\
\varphi_1(x) &= \sqrt{\tfrac{3}{2}}\, x, \\
\varphi_2(x) &= \sqrt{\tfrac{5}{2}} \cdot \tfrac{1}{2}(3x^2 - 1), \\
\varphi_3(x) &= \sqrt{\tfrac{7}{2}} \cdot \tfrac{1}{2}(5x^3 - 3x), \\
\varphi_4(x) &= \sqrt{\tfrac{9}{2}} \cdot \tfrac{1}{8}(35x^4 - 30x^2 + 3), \\
\varphi_5(x) &= \sqrt{\tfrac{11}{2}} \cdot \tfrac{1}{8}(63x^5 - 70x^3 + 15x).
\end{align*}
$$

---

## 6. Estimador de Projeção Ortogonal

O estimador de coeficientes e a função de deriva estimada são (Capítulo 7, Seção 7.3, Equação 7.7):

$$
\hat{\theta}_m = \hat{\Psi}_m^{-1} \hat{Z}_m,
\qquad
\hat{b}_m(x) = \hat{\theta}_m^\top \varphi(x) = \sum_{j=0}^{m-1} \hat{\theta}_{m,j}\, \varphi_j(x)
$$

### Limite de Risco Integrado Não-Adaptativo

Sobre classes de Sobolev \( \mathcal{W}_2^\beta(I) \) (Capítulo 3, Teorema 3.3 e Capítulo 7, Teorema 7.2):

$$
\mathbb{E}\bigl[ \|\hat{b}_m - b_0\|_I^2 \bigr]
\le \min_{b \in \mathcal{S}_m} \|b - b_0\|_I^2 + C \frac{m}{NT}
= O\Bigl( m^{-2\beta} + \frac{m}{NT} \Bigr)
$$

---

## 7. Estabilidade Espectral (Bernstein Matricial)

Para garantir invertibilidade quase certa de \( \hat{\Psi}_m \) aplica-se a Desigualdade de Bernstein Matricial (Capítulo 24, Teorema 24.2):

$$
\mathbb{P}\left( \Bigl\| \sum_{i=1}^{N} Y_i \Bigr\|_{\mathrm{op}} \ge t \right)
\le 2m \exp\left( \frac{-t^2 / 2}{\sigma_{\mathrm{mat}}^2 + L t / 3} \right)
$$

No código utiliza-se regularização ridge:

$$
\hat{\Psi}_m^{\mathrm{reg}} = \hat{\Psi}_m + \varepsilon \cdot \lambda_{\max}(\hat{\Psi}_m)\, I_m,
\qquad \varepsilon = 10^{-5}
$$

---

## 8. U-Estatísticas Degeneradas

Para análise de flutuações de segunda ordem (Capítulo 17, Seções 17.2 e 17.3):

$$
U_N(H_2) = \frac{2}{N(N-1)} \sum_{1 \le i < j \le N} H_2\bigl(X^{(i)}, X^{(j)}\bigr)
$$

com decomposição de Hoeffding:

$$
U_N(H_2) = \theta + H_1(X^{(i)}) + H_1(X^{(j)}) + H_2^{\mathrm{deg}}(X^{(i)}, X^{(j)})
$$

Sob degeneração linear (\( H_1 \equiv 0 \)), a distribuição limite é:

$$
N\, U_N(H_2) \xrightarrow{d} \sum_{k=1}^{\infty} \lambda_k \bigl( Z_k^2 - 1 \bigr)
$$

onde \( Z_k \) são variáveis gaussianas padrão independentes e \( \lambda_k \) são os autovalores do operador integral associado ao núcleo (representação espectral de Mercer).

---

## 9. Risco Integrado \( L^2 \)

O risco de estimação é avaliado pela norma \( L^2 \) integrada sobre uma grade espacial \( \mathcal{G} \) (Capítulo 3, Seção 3.4.4 e Capítulo 7, Seção 7.5):

$$
R_N(\hat{b}_m)
= \int_{\mathcal{G}} \bigl( b_0(x) - \hat{b}_m(x) \bigr)^2\, dx
\approx \mathrm{trapezoid}\bigl( (b_0 - \hat{b}_m)^2, \mathcal{G} \bigr)
$$

---

## 10. Parâmetros Físicos do Nanodispositivo

| Parâmetro                  | Valor                          | Unidade | Descrição                                      |
|---------------------------|--------------------------------|---------|------------------------------------------------|
| Capacitância do canal     | \( 1.0 \times 10^{-15} \)      | F       | Capacitância efetiva do canal                  |
| Frequência de plasma      | \( 1.2 \times 10^{12} \)       | Hz      | Frequência de plasma dos portadores            |
| Temperatura do banho      | \( 1.5 \)                      | K       | Temperatura criogênica de operação             |
| Constante de Boltzmann    | \( 1.380649 \times 10^{-23} \) | J/K     | —                                              |

---

## 11. Instalação e Execução

### Requisitos

```bash
pip install numpy scipy matplotlib
```

### Execução

```bash
python nanodispositivo_pipeline.py
```

O pipeline:

1. Simula \( N = 200 \) trajetórias de comprimento \( n = 1500 \) passos;
2. Constrói a base de Legendre de dimensão \( m = 6 \);
3. Estima \( \hat{b}_m \) via projeção de mínimos quadrados regularizada;
4. Avalia U-estatísticas degeneradas;
5. Calcula o risco \( L^2 \) e gera o gráfico comparativo \( b_0(x) \) vs. \( \hat{b}_m(x) \).

---

## 12. Estrutura do Código

```
nanodispositivo-quantico-pipeline/
├── nanodispositivo_pipeline.py    # Pipeline completo
└── README.md                      # Este arquivo
```

### Classes Principais

| Classe                                   | Responsabilidade                                      |
|------------------------------------------|-------------------------------------------------------|
| `ParametrosFisicosNanodispositivo`       | Constantes físicas e eletrostáticas                   |
| `SimuladorEstocasticoNanodispositivo`    | Simulação de EDEs de Itô (Euler-Maruyama)             |
| `BaseOrthonormalLegendreNano`            | Sistema ortonormal de Legendre                        |
| `EstimadorProjecaoMinimosQuadradosNano`  | Estimador de projeção + regularização espectral       |
| `AnalisadorUStatisticsDegeneradasNano`   | U-estatísticas degeneradas de 2ª ordem                |
| `ValidadorRiscoIntegradoNano`            | Cálculo do risco \( L^2 \) integrado                  |

---

## 13. Referências Teóricas (Wilcke, 2026)

| Capítulo / Seção       | Conteúdo Utilizado                                          |
|------------------------|-------------------------------------------------------------|
| Cap. 3, §3.4.1         | Função objetivo estocástica \( \gamma_N(b) \)               |
| Cap. 3, §3.4.4         | Risco integrado \( L^2 \) e Teorema 3.3                     |
| Cap. 3, §3.5           | Polinômios ortogonais de Jacobi / Legendre                  |
| Cap. 5, §5.1–5.5       | Bases ortonormais e estabilidade espectral                  |
| Cap. 6, §6.2–6.3       | EDEs de Itô e simulação Euler-Maruyama (Eq. 6.4 e 6.9)      |
| Cap. 7, §7.1           | Paradigma de múltiplas trajetórias curtas independentes     |
| Cap. 7, §7.2–7.3       | \( \gamma_N \), \( \hat{\Psi}_m \), \( \hat{Z}_m \), Eq. 7.7 |
| Cap. 7, §7.5           | Limites de risco (Teorema 7.2)                              |
| Cap. 17, §17.2–17.3    | Decomposição de Hoeffding e TLC para U-estatísticas         |
| Cap. 24, §24.6         | Desigualdade de Bernstein matricial (Teorema 24.2)          |

---

## Licença e Citação

Se utilizar este código ou as metodologias implementadas, cite:

```
Wilcke, A. (2026). Métodos Avançados em Inferência Estatística Não-Paramétrica.
Pipeline Nanodispositivo Quântico de Alta Frequência.
```

---

*Repositório preparado para GitHub com todas as equações do modelo renderizadas em LaTeX (macros compatíveis com o MathJax do GitHub).*
