# Inferência em Transistor de Elétron Único (SET)

Sou Luiz Tiago Wilcke. Este repositório contém o código que usei para recuperar, de forma não-paramétrica, a deriva e a volatilidade local de um Single-Electron Transistor a partir de trajetórias simuladas.

A base teórica está nos Capítulos 5, 6, 7 e 18 do meu livro *Métodos Avançados em Inferência Estatística Não-Paramétrica*.

---

## Dinâmica do dispositivo

O estado \(X_t\) segue a EDE de Itô

$$
dX_t = b_0(X_t)\,dt + \sigma_0(X_t)\,dW_t
$$

com

$$
b_0(x) = -2\sin(\pi x) - x, \qquad \sigma_0(x) = 0.4\bigl(1 + 0.5 x^2\bigr).
$$

A discretização é o esquema de Euler-Maruyama:

$$
X_{k+1} = X_k + b_0(X_k)\Delta t + \sigma_0(X_k)\Delta W_k.
$$

---

## Como recupero a deriva

Projeto os incrementos observados sobre a base de Legendre \(\{\varphi_j\}_{j=0}^{m-1}\). A matriz de design empírica e o vetor de momentos são

$$
\hat\Psi_m = \frac{1}{NT}\sum_{i=1}^N\sum_k \varphi(X_{i,k})\varphi(X_{i,k})^\top\Delta t,
\qquad
\hat Z_m = \frac{1}{NT}\sum_{i=1}^N\sum_k \varphi(X_{i,k})\,\Delta X_{i,k}.
$$

Resolvo o sistema regularizado

$$
\bigl(\hat\Psi_m + \varepsilon I\bigr)\hat\theta_m = \hat Z_m
$$

e obtenho

$$
\hat b_m(x) = \sum_{j=0}^{m-1}\hat\theta_j\,\varphi_j(x).
$$

---

## Volatilidade local

Pela identidade de Itô, \(\frac{(\Delta X)^2}{\Delta t}\) observa \(a(X)=\sigma_0^2(X)\). Aplico a mesma projeção:

$$
\hat a_m(x) = \sum_{j=0}^{m-1}\hat\eta_j\,\varphi_j(x),
$$

onde \(\hat\eta\) resolve o sistema análogo construído com a variação quadrática discreta.

---

## Execução

```bash
pip install numpy matplotlib
python modelo_set.py
```

O script gera 1000 trajetórias, ajusta os dois estimadores e mostra os gráficos comparativos.

---

## Classe principal

`ModeloSETCompletoOtimizado` concentra a simulação, a base de Legendre e os dois ajustadores (deriva e volatilidade).

---

Referência: meu livro, Capítulos 5, 6, 7 e 18.
