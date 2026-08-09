# Modelos Não-Paramétricos

**Repositório de Implementações Computacionais de Métodos Avançados em Inferência Estatística Não-Paramétrica**

Este repositório contém frameworks e pipelines computacionais que aplicam a teoria desenvolvida no livro:

> **Wilcke, Luiz Tiago (2026)**  
> *Métodos Avançados em Inferência Estatística Não-Paramétrica*  
> *Teoria Matemática, Processos de Difusão e Estimação por Kernel*  
> Uma Abordagem Teórica e Prática com EDEs, Processos de Markov, Método PCO, Estimadores de Projeção de Mínimos Quadrados e Testes de Aderência

Todos os códigos deste repositório têm fundamentos matemáticos rigorosos baseados no livro acima. Cada pasta implementa um modelo ou framework específico de estimação não-paramétrica aplicado a sistemas físicos reais (sensores, dispositivos quânticos, nanoeletrônica e processos estocásticos).

---

## Adquira o Livro Completo

O livro **Métodos Avançados em Inferência Estatística Não-Paramétrica** (2026) contém toda a teoria matemática utilizada nos códigos deste repositório, com demonstrações, exercícios resolvidos e aplicações.

**Preço: R$ 60,00**

**Pagamento via PIX:**  
`tiagowilcke8@gmail.com`

**Como receber o livro:**

1. Realize o pagamento de R$ 60,00 via PIX para a chave acima.
2. Envie um e-mail para **tiagowilcke8@gmail.com** informando:
   - Nome completo
   - Comprovante do PIX (anexo ou print)
   - E-mail para recebimento do arquivo
3. O livro em PDF será enviado após a confirmação do pagamento.

---

## Estrutura do Repositório

| Pasta | Descrição | Capítulos Principais do Livro |
|-------|-----------|-------------------------------|
| [csnnd-neutrinos-pipeline](./csnnd-neutrinos-pipeline) | Nanosensor Supercondutor para Neutrinos (CSNND). Simulação de EDEs de Itô-Lévy com saltos + estimador de projeção. | Cap. 6, 7, 8, 17, 25 |
| [gqd-sensor-framework](./gqd-sensor-framework) | Framework de sensor baseado em Graphene Quantum Dot (GQD). | Cap. 3, 5, 9 |
| [led-photon-unico-framework](./led-photon-unico-framework) | Framework de detecção de fóton único com LED. | Cap. 6, 9, 11 |
| [nems-fractional-framework](./nems-fractional-framework) | Sistemas Nanoeletromecânicos (NEMS) com difusões fracionárias. | Cap. 8, 25 |
| [quantum-dot-framework](./quantum-dot-framework) | Framework de Quantum Dot para estimação não-paramétrica. | Cap. 3, 5, 7 |
| [set-transistor-framework](./set-transistor-framework) | Framework de Single-Electron Transistor (SET). | Cap. 6, 7, 9 |
| [stno-inference](./stno-inference) | Inferência em Spin-Torque Nano-Oscillator (STNO). | Cap. 7, 8, 17 |

---

## Fundamentos Matemáticos Comuns

Todos os projetos compartilham a base teórica extraída do livro.

### 1. Estimação Não-Paramétrica por Kernel

Estimador de densidade de Rosenblatt-Parzen:

$$
\hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^n K\left( \frac{x - X_i}{h} \right)
$$

Estimador de regressão de Nadaraya-Watson:

$$
\hat{m}_h(x) = \frac{\sum_{i=1}^n K_h(x - X_i) Y_i}{\sum_{i=1}^n K_h(x - X_i)}
$$

### 2. Equações Diferenciais Estocásticas (EDEs)

Processo de difusão clássico de Itô:

$$
dX_t = b(X_t)\, dt + \sigma(X_t)\, dW_t
$$

Processo de Itô-Lévy com saltos:

$$
dX_t = b(X_t)\, dt + \sigma(X_t)\, dW_t + dJ_t
$$

Difusão fracionária (movimento browniano fracionário \( B^H \)):

$$
dX_t = b(X_t)\, dt + \sigma(X_t)\, dB^H_t
$$

### 3. Estimador de Projeção de Mínimos Quadrados

Para a função de deriva \( b \), o estimador no subespaço \( S_m \) é:

$$
\hat{b}_m = \arg\min_{b \in S_m} \gamma_N(b)
$$

com a função objetivo estocástica:

$$
\gamma_N(b) = \frac{1}{NT} \sum_{i=1}^N \int_0^T |b(X_s^{(i)})|^2\, ds - \frac{2}{NT} \sum_{i=1}^N \int_0^T b(X_s^{(i)})\, dX_s^{(i)}
$$

A solução explícita utiliza a matriz de design empírica:

$$
\hat{\Psi}_m = \frac{1}{NT} \sum_{i=1}^N \int_0^T \varphi(X_s^{(i)}) \varphi(X_s^{(i)})^\top\, ds
$$

e o vetor de dados:

$$
\hat{Z}_m = \frac{1}{NT} \sum_{i=1}^N \int_0^T \varphi(X_s^{(i)})\, dX_s^{(i)}
$$

resultando em:

$$
\hat{\theta}_m = \hat{\Psi}_m^{-1} \hat{Z}_m, \qquad \hat{b}_m(x) = \hat{\theta}_m^\top \varphi(x)
$$

### 4. Método PCO (Penalized Comparison to Overfitting)

Seleção adaptativa de largura de banda (Capítulo 4):

$$
\hat{h} = \arg\min_h \left\{ \|\hat{m}_h - \hat{m}_{h_{\min}}\|^2 + \mathrm{pen}(h) \right\}
$$

### 5. U-Estatísticas Degeneradas

Para análise de flutuações de segunda ordem:

$$
U_N(H) = \frac{2}{N(N-1)} \sum_{1 \leq i < j \leq N} H(X^{(i)}, X^{(j)})
$$

com decomposição de Hoeffding e Teorema do Limite Central associado (Capítulo 17).

### 6. Cálculo de Malliavin e Integral de Skorokhod

Isometria de Skorokhod para processos não-adaptados (Capítulo 25):

$$
\mathbb{E}[\delta(u)^2] = \mathbb{E}[\|u\|_{L^2}^2] + \mathbb{E}[\langle Du, Du \rangle_{L^2 \otimes L^2}]
$$

---

## Processos de Lévy (Capítulo 8)

Um processo de Lévy \( X = (X_t)_{t \geq 0} \) possui incrementos independentes e estacionários, com trajetórias càdlàg.

### Teorema de Lévy-Khintchine

A função característica admite a forma:

$$
\mathbb{E}[e^{i u X_t}] = \exp(t \psi(u))
$$

com expoente característico:

$$
\psi(u) = i b u - \frac{1}{2} \sigma^2 u^2 + \int_{\mathbb{R}} \left( e^{i u x} - 1 - i u x \mathbf{1}_{|x|<1} \right) \nu(dx)
$$

onde \( (b, \sigma^2, \nu) \) é a triplet de Lévy.

### Decomposição de Lévy-Itô

$$
X_t = b t + \sigma W_t + \int_{|x|<1} x \tilde{N}(t,dx) + \int_{|x| \geq 1} x N(t,dx)
$$

No pipeline CSNND os saltos são modelados por um processo de Poisson composto:

$$
J_t = \sum_{i=1}^{N_t} \xi_i, \qquad \xi_i \sim \mathcal{N}(0, 0.60^2)
$$

---

## Como usar este repositório

Cada pasta é um projeto independente. Entre na pasta desejada e consulte o `README.md` específico para detalhes das equações, parâmetros e instruções de execução.

Exemplo:

```bash
cd csnnd-neutrinos-pipeline
python csnnd_pipeline.py
```

---

## Dependências Gerais

```bash
pip install numpy scipy matplotlib
```

Alguns frameworks podem exigir pacotes adicionais (indicados no README de cada pasta).

---

## Sobre o Autor e o Livro

**Luiz Tiago Wilcke**

O livro cobre, de forma rigorosa e com exercícios resolvidos:

- **Parte I** – Fundamentos da Estimação Não-Paramétrica (Kernels, redução de viés, seleção de banda via PCO, projeção ortogonal)
- **Parte II** – Inferência em Equações Diferenciais Estocásticas (Itô, saltos, fracionárias, Nadaraya-Watson para EDEs, estimadores de distribuição, testes de aderência)

---

## Citação

```
Wilcke, Luiz Tiago (2026).
Métodos Avançados em Inferência Estatística Não-Paramétrica:
Teoria Matemática, Processos de Difusão e Estimação por Kernel.
```

---

*Todos os modelos deste repositório são implementações computacionais fiéis à teoria desenvolvida no livro.*
