# Inferência em Ponto Quântico com EDEs e Saltos

Sou Luiz Tiago Wilcke. Este código recupera a função de deriva de um ponto quântico a partir de trajetórias simuladas com saltos, usando projeção ortogonal sobre polinômios de Legendre.

A teoria está nos Capítulos 6, 7 e 8 do meu livro *Métodos Avançados em Inferência Estatística Não-Paramétrica* (2026).

---

## Dinâmica

O estado \(X_t\) do ponto quântico obedece

$$
dX_t = b_0(X_t)\,dt + \sigma(X_t)\,dW_t + dJ_t
$$

em que \(J_t\) é um processo de Poisson composto. Usei

$$
b_0(x) = -3x + 0.5x^3, \qquad \sigma(x) = 0.4\bigl(1 + 0.1x^2\bigr).
$$

A discretização é Euler-Maruyama acrescida dos saltos:

$$
X_{k+1} = X_k + b_0(X_k)\Delta t + \sigma(X_k)\Delta W_k + \xi_k.
$$

---

## Estimador de projeção

Monto a matriz de design e o vetor de momentos empíricos

$$
\hat\Psi_m = \frac{1}{NT}\sum_{i=1}^N\sum_k \varphi(X_{i,k})\varphi(X_{i,k})^\top\Delta t,
\qquad
\hat Z_m = \frac{1}{NT}\sum_{i=1}^N\sum_k \varphi(X_{i,k})\,\Delta X_{i,k}.
$$

Resolvo

$$
\bigl(\hat\Psi_m + \varepsilon I\bigr)\hat\theta_m = \hat Z_m
$$

e reconstruo

$$
\hat b_m(x) = \sum_{j=0}^{m-1}\hat\theta_j\,\varphi_j(x)
$$

com \(\varphi_j\) sendo a base de Legendre normalizada.

---

## Como rodar

```bash
pip install numpy matplotlib
python simulador_quantum_dot.py
```

O script gera as trajetórias, ajusta o estimador e mostra o gráfico comparando \(b_0\) verdadeira com \(\hat b_m\).

---

## Classes

- `SimuladorQuantumDotEDEs` — gera as trajetórias com saltos
- `EstimadorProjecaoMinimosQuadrados` — monta \(\hat\Psi_m\), \(\hat Z_m\) e reconstrói a deriva

---

Referência: meu livro, Capítulos 6, 7 e 8.
