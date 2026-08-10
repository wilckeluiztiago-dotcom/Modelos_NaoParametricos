# Simulador de Contágio Sistêmico (Crise Subprime 2007-2008)

**Autor:** Luiz Tiago Wilcke

Este repositório contém a implementação em Python de um modelo estocástico avançado para simulação de contágio financeiro utilizando **Processos de Ponto Autocitados (Processos de Hawkes)**. O código foi desenvolvido para modelar e calibrar a dinâmica de propagação de defaults e perdas sistêmicas durante a crise financeira de 2007-2008.

---

## 📖 Origem Teórica e Fundamentação

As equações, a formulação analítica e os fundamentos teóricos que estruturam este modelo foram retirados do **Capítulo 39 (Processos de Ponto Autocitados e Intensidade Condicional)**

O modelo emprega a teoria de intensidades condicionais para descrever como choques exógenos macroeconômicos e o feedback endógeno (memória de eventos anteriores) interagem para gerar efeitos dominó nos diferentes setores do mercado financeiro.

### Equações do Modelo

A taxa de **Intensidade Condicional** $\lambda_i(t)$ para um setor $i$ é regida pela equação de Hawkes multidimensional:

$$\lambda_i(t) = \mu_i(t) + \sum_{j=1}^{d} \sum_{t_k < t} \alpha_{ij} e^{-\beta_{ij}(t - t_k)}$$

Onde:
- $\lambda_i(t)$: Intensidade instantânea de ocorrência de eventos (defaults/crises) no setor $i$.
- $\mu_i(t)$: Componente exógeno correspondente aos marcos macroeconômicos e históricos da crise.
- $\alpha_{ij}$: Parâmetro de contágio cruzado (magnitude do impacto transmitido do setor $j$ para o setor $i$).
- $\beta_{ij}$: Taxa de resfriamento ou decaimento exponencial da memória do choque.

### Condição de Estabilidade do Sistema

A estabilidade analítica do sistema é validada por meio da **Matriz de Ramificação** ($\Gamma$), cujos elementos são dados por:

$$\Gamma_{ij} = \frac{\alpha_{ij}}{\beta_{ij}}$$

O sistema é considerado estacionário e estável se o raio espectral (o maior autovalor em módulo) da matriz de ramificação satisfizer a condição:

$$\rho(\Gamma) < 1$$

---

## 🛠️ Simulação e Calibração

O código utiliza o **Algoritmo de Thinning de Ogata** para a simulação contínua do processo estocástico, incorporando uma cronologia histórica real dividida em marcos fundamentais:
1. **Abril de 2007**: Falência da New Century (*Subprime*).
2. **Agosto de 2007**: BNP Paribas congela fundos (Liquidez).
3. **Outubro/Novembro de 2007**: Perdas bilionárias em instituições (Citi / Merrill Lynch).
4. **Março de 2008**: Resgate do Bear Stearns.
5. **Setembro de 2008**: Falência do Lehman Brothers e socorro à AIG.

### Setores Modelados
- **Setor 0**: Carteiras Legadas de MBS / CDOs (Subprime).
- **Setor 1**: Bancos de Investimento (Wall Street).
- **Setor 2**: Sistema de Crédito & Seguradoras (AIG).

---

## 🚀 Como Executar

### Pré-requisitos
Certifique-se de ter as bibliotecas `numpy` e `matplotlib` instaladas no seu ambiente Python:

```bash
pip install numpy matplotlib
