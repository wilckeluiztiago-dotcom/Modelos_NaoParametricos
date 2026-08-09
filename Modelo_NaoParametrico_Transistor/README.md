# Framework Estatístico Não-Paramétrico e Estocástico para Nanodispositivos

**Transistor de Átomo Único de Fósforo (Si:P)**

Autor: **Luiz Tiago Wilcke**

---

## Visão Geral

Este repositório implementa um **framework estatístico completo** em Python para modelagem, simulação e estimação não-paramétrica de dinâmicas estocásticas em nanodispositivos, com foco no transistor de átomo único de fósforo em silício (Si:P).

O código cobre:

1. Simulação de Equações Diferenciais Estocásticas (EDEs) de **Itô-Lévy** com saltos (modelo de Random Telegraph Signal – RTS)
2. Estimação não-paramétrica da função de **deriva** via **Nadaraya-Watson**
3. Seleção adaptativa de largura de banda pelo critério **PCO** (Penalized Comparison to Overfitting)
4. Estimação da **volatilidade local** por variação quadrática
5. Análise de **confiabilidade** através da função de **Hazard** suavizada

---

## Modelo Teórico

### 1. Equação Diferencial Estocástica de Itô-Lévy

O estado do sistema \(X_t\) (posição efetiva / carga no poço quântico) é governado por:

$$
dX_t = b(X_t)\,dt + \sigma\,dW_t + dJ_t
$$

onde:

- \(b(x)\) é a função de deriva (força do potencial)
- \(\sigma\) é a volatilidade (ruído térmico/quântico)
- \(W_t\) é um movimento browniano padrão
- \(J_t\) é um processo de Poisson composto que modela os saltos do **Random Telegraph Signal (RTS)**

#### Função de deriva teórica (poço de potencial assimétrico)

$$
b_0(x) = -2.5\,x\,\exp(-0.4\,x^2) + 0.3\,x
$$

#### Discretização (Esquema de Euler-Maruyama com saltos)

$$
X_{t+\Delta t} = X_t + b(X_t)\,\Delta t + \sigma\,\Delta W_t + \xi_t
$$

com \(\Delta W_t \sim \mathcal{N}(0,\Delta t)\) e \(\xi_t\) sendo o salto de Poisson composto:

$$
\xi_t = \mathbf{1}_{\{U < \lambda\Delta t\}} \cdot Z,\qquad Z\sim\mathcal{N}(0,0.35^2)
$$

---

### 2. Estimador de Nadaraya-Watson para a Deriva

A partir de \(N\) trajetórias discretizadas, o estimador não-paramétrico da deriva é:

$$
\hat{b}_h(x) = \frac{\displaystyle\sum_{i=1}^{N}\sum_{k=0}^{n-1} K_h(X_{i,k}-x)\,\Delta X_{i,k}}{\displaystyle\sum_{i=1}^{N}\sum_{k=0}^{n-1} K_h(X_{i,k}-x)\,\Delta t}
$$

onde \(K_h(u) = \frac{1}{h}K\left(\frac{u}{h}\right)\) e \(K\) é o **kernel de Epanechnikov**:

$$
K(u) = \frac{3}{4}(1-u^2)\,\mathbf{1}_{|u|\le 1}
$$

---

### 3. Seleção Adaptativa de Largura de Banda — PCO

O método **Penalized Comparison to Overfitting (PCO)** escolhe automaticamente a largura de banda \(h\) minimizando o critério:

$$
\text{Critério}_{\text{PCO}}(h) = R(\hat{b}_h,\hat{b}_{h_{\min}}) + \text{Pen}(h)
$$

com o termo de contraste

$$
R(\hat{b}_h,\hat{b}_{h_{\min}}) = \sum_{x\in\mathcal{G}} \bigl(\hat{b}_h(x) - \hat{b}_{h_{\min}}(x)\bigr)^2
$$

e a penalização teórica

$$
\text{Pen}(h) = \frac{2}{N h} - \frac{0.01}{h}
$$

---

### 4. Estimação da Volatilidade Local

A função de volatilidade local \(a(x)=\sigma^2(x)\) é estimada via variação quadrática suavizada:

$$
\hat{a}_h(x) = \frac{\displaystyle\sum_{i,k} K_h(X_{i,k}-x)\,\frac{(\Delta X_{i,k})^2}{\Delta t}}{\displaystyle\sum_{i,k} K_h(X_{i,k}-x)}
$$

(utilizando kernel gaussiano no código)

---

### 5. Função de Hazard (Análise de Confiabilidade)

A taxa de falha instantânea (hazard) é estimada de forma não-paramétrica:

$$
\hat{\lambda}_h(t) = \frac{\hat{f}_h(t)}{\hat{S}_h(t)}
$$

onde \(\hat{f}_h\) é a densidade kernel e \(\hat{S}_h\) é a função de sobrevivência suavizada.

---

## Estrutura do Código

| Classe | Responsabilidade |
|--------|------------------|
| `SimuladorEDELevyTransistor` | Simulação de trajetórias Itô-Lévy com saltos RTS |
| `EstimadorNadarayaWatsonEDEs` | Estimação não-paramétrica da deriva |
| `SelecaoAdaptativaPCO` | Seleção automática da largura de banda \(h\) |
| `EstimadorVolatilidadeLocal` | Estimação de \(a(x)=\sigma^2(x)\) |
| `AnaliseConfiabilidadeHazard` | Estimação da função de hazard |

---

## Requisitos

```bash
pip install numpy scipy
```

---

## Como Executar

```bash
python simulador_sip.py
```

O script executa automaticamente:

1. Simulação de 250 trajetórias
2. Seleção adaptativa de \(h\) via PCO
3. Estimação da deriva e comparação com a verdade teórica
4. Estimação da volatilidade local
5. Análise de confiabilidade (hazard)

---

## Exemplo de Saída

```
=====================================================================
 MODELO ESTATÍSTICO COMPLETO: TRANSISTOR DE FÓSFORO (Si:P)
 Autor: Luiz Tiago Wilcke
=====================================================================

[*] Simulando 250 trajetórias estocásticas de Itô-Lévy...
[*] Executando seleção adaptativa de largura de banda via PCO...
[+] Largura de banda ótima selecionada via PCO: h_opt = 0.XXXX

--- RESULTADOS DA DERIVA ESTIMADA (Adaptativa PCO) ---
...
--- RESULTADOS DA VOLATILIDADE LOCAL ---
...
--- RESULTADOS DE CONFIABILIDADE (Função de Hazard) ---
...
[+] Execução computacional do modelo estatístico encerrada com êxito!
```

---

## Referências Teóricas (Capítulos citados no código)

- Cap. 4  – Seleção de largura de banda por PCO  
- Cap. 6 e 8 – EDEs de Itô-Lévy e saltos  
- Cap. 9  – Estimador de Nadaraya-Watson para deriva  
- Cap. 15 – Função de Hazard suavizada  
- Cap. 18 – Volatilidade local via variação quadrática  

---

## Licença

Este projeto é disponibilizado para fins acadêmicos e de pesquisa.  
Sinta-se livre para usar, modificar e citar.

---

**Autor:** Luiz Tiago Wilcke  
**Projeto:** Framework Estatístico Não-Paramétrico e Estocástico Completo para Nanodispositivos (Si:P)
