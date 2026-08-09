# Inferência em NEMS com Memória Fracionária

Este código simula um nanoeletromecânico (NEMS) dirigido por movimento browniano fracionário e recupera a deriva por projeção ortogonal. Também calcula uma U-estatística degenerada de segunda ordem como diagnóstico de dispersão.

A teoria está nos Capítulos 3, 5, 6, 7, 8 e 17 do meu livro *Métodos Avançados em Inferência Estatística Não-Paramétrica* (2026).

---

## Movimento browniano fracionário

O processo \(B^H_t\) com parâmetro de Hurst \(H\in(0,1)\) tem covariância

$$
\mathbb{E}\bigl[B^H_t B^H_s\bigr]
=
\frac12\Bigl(t^{2H}+s^{2H}-|t-s|^{2H}\Bigr).
$$

No código a matriz de Gram é montada nessa fórmula e fatorada por Cholesky:

$$
\Sigma = LL^\top,\qquad
B^H = Lz,\quad z\sim\mathcal{N}(0,I).
$$

---

## Dinâmica do oscilador

A posição \(X_t\) obedece a EDE fracionária

$$
dX_t = b_0(X_t)\,dt + \sigma\,dB^H_t
$$

com força restauradora cúbica

$$
b_0(x) = -2.5x - x^3
$$

e difusão constante \(\sigma=0.4\). A discretização integrada ao caminho do mBf fica

$$
X_{k+1}
=
X_k + b_0(X_k)\Delta t_k + \sigma\,\Delta B^H_k,
\qquad
\Delta B^H_k = B^H_{t_{k+1}}-B^H_{t_k}.
$$

---

## Estimador de projeção

A base \(\{\varphi_j\}_{j=0}^{m-1}\) é a família de Legendre normalizada. A matriz de design empírica e o vetor de momentos são

$$
\hat\Psi_m
=
\frac1{NT}
\sum_{i=1}^N\sum_{k=0}^{n-1}
\varphi(X_{i,k})\varphi(X_{i,k})^\top\Delta t,
$$

$$
\hat Z_m
=
\frac1{NT}
\sum_{i=1}^N\sum_{k=0}^{n-1}
\varphi(X_{i,k})\,\Delta X_{i,k}.
$$

O sistema regularizado

$$
\bigl(\hat\Psi_m + \varepsilon I\bigr)\hat\theta_m = \hat Z_m
$$

é resolvido e a deriva recuperada é

$$
\hat b_m(x)
=
\sum_{j=0}^{m-1}\hat\theta_j\,\varphi_j(x).
$$

O menor autovalor de \(\hat\Psi_m\) é impresso como diagnóstico de estabilidade espectral.

---

## U-estatística degenerada

Para medir dispersão entre trajetórias, formo o núcleo simétrico de segunda ordem

$$
H_2(X^i,X^j)
=
\frac1n\sum_{k=0}^{n}
\bigl(X^i_k - X^j_k\bigr)^2
$$

e a U-estatística amostral

$$
U_N
=
\binom{N'}{2}^{-1}
\sum_{1\le i<j\le N'}
H_2(X^i,X^j),
$$

com \(N'=\min(N,20)\) para controle de custo.

---

## Como rodar

```bash
pip install numpy matplotlib scipy
python simulador_nems_fractional.py
```

O script gera as trajetórias fracionárias, ajusta a projeção, imprime o autovalor mínimo e a U-estatística, e mostra o gráfico de \(b_0\) versus \(\hat b_m\).

---

## Classes

| Classe | Função |
|--------|--------|
| `SimuladorNEMSFractional` | Covariância do mBf, Cholesky e integração da fSDE |
| `EstimadorProjecaoMalliavinNEMS` | Monta \(\hat\Psi_m\), \(\hat Z_m\) e reconstrói a deriva |
| `EvaluadorUStatisticsRisco` | Calcula \(U_N(H_2)\) |

---

Referência: meu livro, Capítulos 3, 5, 6, 7, 8 e 17.
