# Inferência Não-Paramétrica em Sensor de Grafeno (GQD)

Código desenvolvido para estudar a dinâmica de um sensor quântico de grafeno (Graphene Quantum Dot) sob equações diferenciais estocásticas com saltos.

O material parte da obra *Métodos Avançados em Inferência Estatística Não-Paramétrica* e concentra três blocos principais: geração de trajetórias Itô-Lévy, recuperação da deriva por projeção ortogonal e recuperação da volatilidade local a partir da variação quadrática.

---

## O que o programa faz

1. Gera várias trajetórias curtas e independentes de uma difusão com saltos (modelo de armadilha / Random Telegraph Signal).
2. Estima a função de deriva \(b_0\) projetando o incremento observado sobre uma base de Legendre.
3. Estima a volatilidade local \(a(x)=\sigma^2(x)\) suavizando a variação quadrática discreta.

---

## Dinâmica considerada

O estado \(X_t\) do sensor obedece

$$
dX_t = b_0(X_t)\,dt + \sigma_0(X_t)\,dW_t + dJ_t
$$

em que \(J_t\) é um processo de Poisson composto. No código a deriva e a difusão são dadas por

$$
b_0(x) = -2.8\tanh(x) - 1.2x, \qquad \sigma_0(x) = 0.4 + 0.1x^2.
$$

A discretização empregada é o esquema de Euler-Maruyama acrescido dos saltos:

$$
X_{k+1} = X_k + b_0(X_k)\Delta t + \sigma_0(X_k)\Delta W_k + \xi_k,
$$

com \(\Delta W_k\sim\mathcal{N}(0,\Delta t)\) e \(\xi_k\) nulo com alta probabilidade ou gaussiano quando o salto ocorre.

---

## Estimação da deriva por projeção

Seguindo a abordagem dos Capítulos 3, 5 e 7, procura-se o vetor de coeficientes \(\hat\theta_m\) que minimiza o risco empírico contínuo. Define-se a matriz de Gram empírica e o vetor de momentos

$$
\Psi_m = \frac{1}{NT}\sum_{i=1}^N\sum_{k}\varphi(X_{i,k})\varphi(X_{i,k})^\top\Delta t,
\qquad
Z_m = \frac{1}{NT}\sum_{i=1}^N\sum_{k}\varphi(X_{i,k})\,\Delta X_{i,k}.
$$

O sistema

$$
\Psi_m\hat\theta_m = Z_m
$$

é resolvido após uma regularização de Tikhonov leve. A base \(\varphi_j\) utilizada é a família de polinômios de Legendre normalizados. A função recuperada fica

$$
\hat b_m(x) = \sum_{j=0}^{m-1}\hat\theta_j\,\varphi_j(x).
$$

---

## Volatilidade local

A partir da identidade de Itô, a variação quadrática normalizada

$$
\frac{(\Delta X_{i,k})^2}{\Delta t}
$$

serve como observação ruidosa de \(a(X_{i,k})\). Um suavizador gaussiano produz a estimativa pontual

$$
\hat a_h(x) = \frac{\sum_{i,k}K_h(X_{i,k}-x)\frac{(\Delta X_{i,k})^2}{\Delta t}}{\sum_{i,k}K_h(X_{i,k}-x)}.
$$

---

## Como rodar

```bash
pip install numpy
python simulador_gqd.py
```

O script imprime, lado a lado, os valores teóricos e estimados da deriva e da volatilidade em uma grade espacial.

---

## Organização das classes

| Classe | Papel |
|--------|-------|
| `SimuladorGQDAvancado` | Gera as trajetórias com saltos |
| `ProjecaoMinimosQuadradosEDEs` | Monta \(\Psi_m\), \(Z_m\) e reconstrói \(\hat b_m\) |
| `EstimadorVolatilidadeLocalEDEs` | Suaviza a variação quadrática |

---

## Referência

Wilcke, L. T. *Métodos Avançados em Inferência Estatística Não-Paramétrica*. Capítulos 3, 5, 6, 7, 8 e 18.

---

Qualquer dúvida sobre a implementação ou sobre a teoria por trás, mande um e-mail para wilckeluiztiago@gmail.com 
