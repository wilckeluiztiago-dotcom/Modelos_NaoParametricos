# Inferência Não-Paramétrica em EDEs para Osciladores Nanoeletromagnéticos de Spin (STNO)

**Autor:** Luiz Tiago Wilcke  

Implementação computacional do estimador de projeção ortogonal para a função de deriva de equações diferenciais estocásticas de Itô, aplicada a um oscilador nanoeletromagnético de spin (STNO). A fundamentação teórica, os esquemas de simulação, as matrizes de design empíricas, a análise espectral e as U-estatísticas degeneradas seguem integralmente o livro *Métodos Avançados em Inferência Estatística Não-Paramétrica* (Wilcke, 2026).

---

## Modelo Estocástico

O estado angular $x_t$ da camada livre obedece à EDE de Itô

$$
\mathrm{d}x_t = b_0(x_t)\,\mathrm{d}t + \sigma_0(x_t)\,\mathrm{d}W_t,
$$

onde $W_t$ é um movimento browniano padrão unidimensional.

### Função de deriva (torque de Słonczewski + amortecimento)

$$
b_0(x) = -2.5\sin(x) + 1.2\sin(2x) - 0.6x.
$$

### Coeficiente de difusão local

$$
\sigma_0(x) = 0.40\bigl(1 + 0.15\cos^2(x)\bigr).
$$

### Discretização de Euler–Maruyama

Dado passo temporal $\Delta t = T/n$,

$$
x_{k+1} = x_k + b_0(x_k)\Delta t + \sigma_0(x_k)\sqrt{\Delta t}\,\xi_k, \qquad \xi_k\sim\mathcal{N}(0,1).
$$

São geradas $N$ trajetórias independentes de comprimento $T$, conforme o paradigma de múltiplas trajetórias curtas (Cap. 7, Seç. 7.1).

---

## Base ortonormal de Legendre

Sobre o intervalo rescalado $[-1,1]$, os polinômios de Legendre normalizados $\phi_j$ formam o sistema

$$
\phi_0(u)=\sqrt{\tfrac12},\quad
\phi_1(u)=\sqrt{\tfrac32}\,u,\quad
\phi_2(u)=\sqrt{\tfrac52}\cdot\tfrac12(3u^2-1),\quad\ldots
$$

com $u=x/\pi$. O espaço de aproximação de dimensão $m$ é gerado por $\{\phi_0,\dots,\phi_{m-1}\}$.

---

## Estimador de projeção de mínimos quadrados

A função objetivo empírica (Cap. 3, Seç. 3.4.1) é minimizada projetando $b_0$ sobre o subespaço gerado pela base. As quantidades empíricas contínuas são

$$
\hat\Psi_m = \frac1{NT}\sum_{i=1}^N\sum_{k=0}^{n-1}\phi(x_{i,k})\phi(x_{i,k})^\top\Delta t,
$$

$$
\hat Z_m = \frac1{NT}\sum_{i=1}^N\sum_{k=0}^{n-1}\phi(x_{i,k})\,\Delta x_{i,k}.
$$

O vetor de coeficientes $\hat\theta_m$ resolve o sistema linear regularizado

$$
\bigl(\hat\Psi_m + \lambda I_m\bigr)\hat\theta_m = \hat Z_m,
$$

onde $\lambda$ é escolhido proporcional ao maior autovalor de $\hat\Psi_m$ para garantir estabilidade espectral. A estimativa da deriva fica

$$
\hat b_m(x) = \hat\theta_m^\top\phi(x).
$$

---

## U-estatística degenerada de segunda ordem

Para quantificar flutuações residuais entre trajetórias utiliza-se o núcleo

$$
H_2(X^i,X^j) = \overline{(X^i-X^j)^2}\cdot\exp\bigl(-\tfrac12\mathrm{Var}(X^i-X^j)\bigr).
$$

A U-estatística correspondente (Cap. 17) é

$$
U_N(H_2)=\binom{N}{2}^{-1}\sum_{1\le i<j\le N}H_2(X^i,X^j).
$$

---

## Risco integrado $L^2$

A qualidade da reconstrução é medida por

$$
R_N(\hat b_m)=\int_{-\pi/2}^{\pi/2}\bigl(b_0(x)-\hat b_m(x)\bigr)^2\,\mathrm{d}x.
$$

---

## Estrutura do código

| Classe / Função                        | Papel principal                                      |
|----------------------------------------|------------------------------------------------------|
| `ParametrosFisicosSTNO`                | Constantes spintrônicas de referência                |
| `SimuladorEstocasticoEDEs`             | Geração de trajetórias via Euler–Maruyama            |
| `BaseOrthonormalLegendre`              | Avaliação da base de Legendre                        |
| `EstimadorProjecaoMinimosQuadrados`    | Construção de $\hat\Psi_m$, $\hat Z_m$ e resolução   |
| `AnalisadorUStatisticsDegeneradas`     | Cálculo de $U_N(H_2)$                                |
| `ValidadorRiscoIntegrado`              | Avaliação do risco $L^2$                             |
| `executar_pipeline_completo_stno`      | Orquestra simulação, estimação e visualização        |

---

## Dependências

```
numpy
scipy
matplotlib
```

Instalação rápida:

```bash
pip install numpy scipy matplotlib
```

---

## Execução

```bash
python stno_projection_estimator.py
```

O script imprime o espectro de $\hat\Psi_m$, o valor da U-estatística e o risco $L^2$, além de gerar o gráfico comparativo entre $b_0$ e $\hat b_m$.

---

## Referência teórica

Wilcke, L. T. (2026). *Métodos Avançados em Inferência Estatística Não-Paramétrica*.  
Capítulos 3, 5, 6, 7 e 17 fornecem a base matemática completa utilizada na implementação.
