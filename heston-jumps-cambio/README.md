# Simulação de Risco Cambial com Modelo de Heston e Saltos Co-integrados

Implementação computacional do modelo de difusão com volatilidade estocástica e saltos multiplicativos aplicado à taxa de câmbio USD/BRL, conforme desenvolvido no Capítulo 37 do livro *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel* (Wilcke, 2026).

**Autor:** Luiz Tiago Wilcke

## Descrição

O código realiza:

1. Download de dados históricos reais do par USD/BRL via Yahoo Finance.
2. Detecção de saltos extremos (choques geopolíticos) na série de retornos logarítmicos.
3. Estimação da volatilidade realizada como proxy da variância estocástica.
4. Simulação Monte Carlo de 1.000 trajetórias futuras em espaço logarítmico sob a medida física (risco real), utilizando o processo de Heston com saltos de Poisson compostos.

A simulação é conduzida integralmente em log-preços, o que elimina a possibilidade de preços negativos e preserva a natureza multiplicativa dos saltos.

## Modelo Matemático

### Processo de preço com saltos

Seja \( S_t \) a taxa de câmbio. O modelo é definido no espaço logarítmico:

$$
d(\log S_t) = \left( \mu - \frac{1}{2} v_t \right) dt + \sqrt{v_t}\, dW_t^{(1)} + dJ_t
$$

onde:

- \( \mu \) é o drift histórico anualizado (medida física),
- \( v_t \) é a variância estocástica,
- \( W_t^{(1)} \) é um movimento browniano padrão,
- \( J_t \) é o processo de saltos.

A reconstrução do preço é obtida por:

$$
S_t = \exp(\log S_t)
$$

### Dinâmica da volatilidade estocástica (Heston)

A variância \( v_t \) segue o processo de Cox–Ingersoll–Ross (CIR):

$$
dv_t = \kappa (\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW_t^{(2)}
$$

com a condição de reflexão em zero implementada numericamente:

$$
v_t \leftarrow \max(v_t, 10^{-8})
$$

Os brownianos são correlacionados:

$$
d\langle W^{(1)}, W^{(2)} \rangle_t = \rho\, dt
$$

Na discretização, a correlação é gerada por:

$$
Z_2 = \rho Z_1 + \sqrt{1 - \rho^2}\, Z_{\text{indep}}
$$

### Processo de saltos

Os saltos seguem um processo de Poisson composto:

$$
dJ_t = \sum_{i=1}^{N_t} Y_i
$$

onde \( N_t \) é um processo de Poisson de intensidade \( \lambda \) e os tamanhos dos saltos \( Y_i \) são i.i.d. com distribuição normal:

$$
Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)
$$

No esquema de Euler–Maruyama, a intensidade diária é \( \lambda\, dt \) e os saltos são adicionados diretamente ao incremento do log-preço.

### Detecção empírica de saltos

Um retorno diário \( r_t \) é classificado como salto quando:

$$
|r_t - \bar{r}| > c \cdot \hat{\sigma}
$$

com limiar \( c = 2{,}55 \) e \( \hat{\sigma} \) o desvio-padrão amostral dos retornos logarítmicos. A intensidade anualizada é estimada por:

$$
\hat{\lambda} = \frac{\#\{\text{saltos}\}}{T} \times 252
$$

### Volatilidade realizada

A proxy de alta frequência da volatilidade anualizada é calculada por janela móvel de 21 dias:

$$
\hat{\sigma}_t^{\text{real}} = \sqrt{252} \cdot \operatorname{Std}\{r_{t-20},\dots,r_t\}
$$

A variância inicial e o nível de longo prazo são obtidos a partir dessa série:

$$
v_0 = (\hat{\sigma}_T^{\text{real}})^2, \qquad \theta = \mathbb{E}\bigl[(\hat{\sigma}_t^{\text{real}})^2\bigr]
$$

### Esquema de discretização (Euler–Maruyama)

Com passo de tempo \( \Delta t = 1/252 \):

$$
\begin{aligned}
v_{t+\Delta t} &= \max\Bigl( v_t + \kappa(\theta - v_t)\Delta t + \xi\sqrt{v_t\Delta t}\, Z_2,\; 10^{-8}\Bigr) \\[1em]
\log S_{t+\Delta t} &= \log S_t + \bigl(\mu - \tfrac{1}{2} v_t\bigr)\Delta t + \sqrt{v_t\Delta t}\, Z_1 + Y\cdot\mathbf{1}_{\{N \ge 1\}}
\end{aligned}
$$

onde \( Z_1, Z_2 \) são normais padrão correlacionadas e \( N \sim \operatorname{Poisson}(\lambda\Delta t) \).

## Parâmetros utilizados

| Parâmetro | Símbolo | Valor | Origem |
|-----------|---------|-------|--------|
| Drift histórico | \( \mu \) | média dos retornos × 252 | Dados reais |
| Velocidade de reversão | \( \kappa \) | 3,0 | Calibrado |
| Variância de longo prazo | \( \theta \) | média de \( (\hat{\sigma}^{\text{real}})^2 \) | Dados reais |
| Volatilidade da volatilidade | \( \xi \) | 0,3 | Calibrado |
| Correlação | \( \rho \) | −0,6 | Calibrado |
| Intensidade de saltos | \( \lambda \) | frequência empírica × 252 | Dados reais |
| Média dos saltos | \( \mu_J \) | 0,0 | Fixo |
| Desvio dos saltos | \( \sigma_J \) | 0,02 | Fixo |
| Horizonte | — | 252 dias úteis | — |
| Número de trajetórias | — | 1.000 | — |

## Requisitos

```bash
pip install numpy pandas yfinance matplotlib
```

## Execução

```bash
python simulacao_heston_saltos_cambio.py
```

O script baixa os dados, identifica os saltos, estima a volatilidade realizada, executa a simulação Monte Carlo e gera dois gráficos:

- Série histórica da taxa de câmbio com os saltos destacados.
- Nuvem de trajetórias simuladas e cenário médio projetado para o horizonte de um ano.

## Referência bibliográfica

Wilcke, L. T. (2026). *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel*. Capítulo 37 – Inferência Estatística em Modelos de Difusão com Volatilidade Estocástica e Saltos Co-integrados.

## Licença

Código disponibilizado para fins acadêmicos e de pesquisa.
