# Inferência em LED de Fóton Único (SPS)

Este código recupera a deriva de um single-photon source a partir de trajetórias simuladas, usando projeção ortogonal sobre polinômios de Legendre.

A base teórica está nos Capítulos 6 e 7 do meu livro *Métodos Avançados em Inferência Estatística Não-Paramétrica* (2026).

---

## Dinâmica dos portadores

O estado \(X_t\) (população / intensidade) segue

$$
dX_t = b_0(X_t)\,dt + \sigma(X_t)\,dW_t
$$

com

$$
b_0(x) = -2x + \sin(\pi x), \qquad \sigma(x) = 0.4\bigl(1 + 0.2x^2\bigr).
$$

A discretização usada é o esquema de Euler-Maruyama:

$$
X_{k+1} = X_k + b_0(X_k)\Delta t + \sigma(X_k)\Delta W_k.
$$

---

## Estimador de projeção

Monto a matriz de design e o vetor de momentos

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
\hat b_m(x) = \sum_{j=0}^{m-1}\hat\theta_j\,\varphi_j(x),
$$

em que \(\varphi_j\) são os polinômios de Legendre normalizados.

---

## Como rodar

```bash
pip install numpy matplotlib
python simulador_led_sps.py
```

O script gera as trajetórias, ajusta o estimador e exibe o gráfico comparando \(b_0\) verdadeira com \(\hat b_m\).

---

## Classes

- `SimuladorLEDPhotonUnico` — gera as trajetórias
- `EstimadorProjecaoDerivaLED` — monta \(\hat\Psi_m\), \(\hat Z_m\) e reconstrói a deriva

---

Referência: meu livro, Capítulos 6 e 7.
