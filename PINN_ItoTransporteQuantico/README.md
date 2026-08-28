# PINN-Itô-Transporte-Quântico

**Estimação de Deriva Balística Quântica via Redes Neurais Físico-Informadas e Cálculo de Itô em Nanotransistores GAAFET 3D Sub-3 nm**

**Autor:** Luiz Tiago Wilcke  
**Versão:** 1.0.0  
**Ano:** 2026  
**Base Teórica:** *Métodos Avançados em Inferência Estatística Não-Paramétrica: Teoria Matemática, Processos de Difusão e Estimação por Kernel* (Wilcke, 2026) — Capítulos 7, 9, 25, 28 e 36.

---

## Resumo

Este framework implementa uma **Rede Neural Físico-Informada acoplada ao Cálculo Estocástico de Itô (PINN-Itô)** para modelar o transporte eletrônico quase-balístico em nanotransistores tridimensionais de porta envolvente (**GAAFET 3D – Gate-All-Around Field-Effect Transistors**) em escala sub-3 nm.

O modelo resolve de forma unificada:

1. A equação de Schrödinger com confinamento bidimensional transversal em sub-bandas discretas.
2. A identificação não-paramétrica da função de deriva (*drift*) do quase-nível eletroquímico de Fermi \(\mu(x, t)\) a partir do funcional de contraste empírico de trajetórias estocásticas de Itô (\(\gamma_N(b)\)).
3. A extração das métricas de transporte balístico de Landauer-Büttiker.

O método permite quantificar a velocidade de injeção balística, a transmissão quântica e a corrente balística sem recorrer a simulações custosas de Monte Carlo quântico em malhas volumétricas.

---

## Fundamentação Matemática (referência ao livro)

### 1. Confinamento Transversal e Sub-bandas (Cap. 36 + física de semicondutores)

No canal nanométrico retangular de seção \(W_{\text{nw}} \times H_{\text{nw}}\):

\[
-\frac{\hbar^2}{2 m_{\text{eff}}} \left( \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} \right) \phi_{n,m}(y, z) = E_{n,m} \phi_{n,m}(y, z)
\]

Energia da primeira sub-banda fundamental:

\[
E_{1,1} = \frac{\hbar^2 \pi^2}{2 m_{\text{eff}}} \left( \frac{1}{W_{\text{nw}}^2} + \frac{1}{H_{\text{nw}}^2} \right)
\]

### 2. Equação Diferencial Estocástica de Itô (Cap. 6 e 7)

\[
d\mu_t = b_0(\mu_t)\, dt + \sigma(\mu_t)\, dW_t
\]

onde \(\sigma(\mu) = \sqrt{\frac{2 k_B T \mu_n}{q}}\).

### 3. Funcional de Contraste Empírico (Cap. 7 – Equação 7.2)

\[
\gamma_N(b) = \frac{1}{N T} \sum_{i=1}^N \left( \int_0^T b(\mu_s^i)^2 \, ds - 2 \int_0^T b(\mu_s^i) \, d\mu_s^i \right)
\]

Pela isometria de Itô:

\[
\mathbb{E}[\gamma_N(b)] = |b - b_0|_f^2 - |b_0|_f^2
\]

### 4. Função de Perda Multiobjetivo da PINN-Itô (Cap. 36)

\[
\mathcal{L}_{\text{total}}(\theta) = w_{\text{quant}} \mathcal{L}_{\text{Schrödinger}}(\theta) + w_{\text{Itô}} \gamma_N(b_\theta) + w_{\text{cont}} \left[ \mathcal{L}_{\text{Fonte}}(\theta) + \mathcal{L}_{\text{Dreno}}(\theta) \right]
\]

### 5. Métricas Balísticas

- Velocidade de injeção: \( v_{\text{drift}}(x) = \sqrt{\frac{2 q \, u_\theta(x)}{m_{\text{eff}}}} \)
- Transmissão Landauer-Büttiker (aproximação WKB)
- Corrente balística de Landauer

---

## Estrutura do Projeto (23 Módulos)

```
PINN_ItoTransporteQuantico/
├── README.md
├── __init__.py
├── 01_constantes_fisicas.py
├── 02_parametros_dispositivo.py
├── 03_subbandas_quanticas.py
├── 04_equacao_schrodinger.py
├── 05_processo_ito.py
├── 06_funcional_contraste.py
├── 07_estimador_projecao_minimos_quadrados.py
├── 08_rede_neural_pinn.py
├── 09_perda_multiobjetivo.py
├── 10_treinamento_pinn_ito.py
├── 11_simulacao_trajetorias.py
├── 12_selecao_banda_pco.py
├── 13_kernel_suavizacao.py
├── 14_velocidade_balistica.py
├── 15_transmissao_landauer.py
├── 16_corrente_balistica.py
├── 17_extracao_metricas.py
├── 18_visualizacao_resultados.py
├── 19_validacao_numerica.py
├── 20_utilitarios_matematicos.py
├── 21_configuracao_experimento.py
├── 22_main_execucao.py
└── 23_testes_integracao.py
```

Cada módulo contém aproximadamente 200 linhas de código complexo, com classes, métodos documentados, implementações matemáticas fiéis ao livro e variáveis em português.

---

## Instalação

```bash
pip install torch numpy scipy matplotlib pandas seaborn tqdm
```

Requisitos mínimos:
- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, SciPy, Matplotlib

---

## Como Executar

```bash
cd PINN_ItoTransporteQuantico
python 22_main_execucao.py
```

Ou, de forma modular:

```python
from configuracao_experimento import ConfiguracaoExperimento
from treinamento_pinn_ito import TreinadorPINNIto
from extracao_metricas import ExtratorMetricas

config = ConfiguracaoExperimento()
treinador = TreinadorPINNIto(config)
modelo_treinado = treinador.treinar()
metricas = ExtratorMetricas(modelo_treinado, config)
metricas.gerar_relatorio_completo()
```

---

## Parâmetros do Dispositivo (padrão)

| Parâmetro              | Valor          | Unidade |
|------------------------|----------------|---------|
| Comprimento do canal L | 12             | nm      |
| Largura W              | 3              | nm      |
| Altura H               | 3              | nm      |
| Tensão de Dreno V_D    | 0.65           | V       |
| Tensão de Porta V_G    | 0.70           | V       |
| Temperatura T          | 300            | K       |
| Massa efetiva m_eff    | 0.26 m₀        | -       |

---

## Resultados Esperados

A simulação produz a tabela:

| Posição x (nm) | Quase-Fermi μ(x) (V) | Velocidade Balística (10⁷ cm/s) | Transmissão Landauer T |
|----------------|----------------------|---------------------------------|------------------------|
| 0.00           | 0.0000               | 0.0000                          | 0.0124                 |
| 1.20           | 0.0482               | 0.2541                          | 0.0841                 |
| ...            | ...                  | ...                             | ...                    |
| 12.00          | 0.6500               | 0.9341                          | 1.0000                 |

Além de gráficos de μ(x), velocidade balística, transmissão e mapas de densidade de probabilidade das trajetórias de Itô.

---

## Referências Principais do Livro

1. **Capítulo 7** – Estimação de Deriva via Mínimos Quadrados de Projeção em EDEs (funcional γ_N(b), matriz de design Ψ̂_m, estimador de projeção).
2. **Capítulo 9** – Estimador Nadaraya-Watson para a função de deriva de EDEs.
3. **Capítulo 25** – Cálculo de Malliavin e Integral de Skorokhod.
4. **Capítulo 28** – Classes de Nikol’skii e consistência minimax.
5. **Capítulo 36** – Aprendizado de Máquina Físico-Informado (PINNs) em Equações Diferenciais.

Outras referências clássicas:
- Datta, S. (2005). *Quantum Transport: Atom to Transistor*.
- Natori, K. (1994). Ballistic MOSFET.
- Marie, N. (2025). Nonparametric estimation of the drift in diffusion models.

---

## Licença e Citação

Este código é parte da implementação computacional do livro de Luiz Tiago Wilcke (2026).

```bibtex
@book{wilcke2026,
  author    = {Wilcke, Luiz Tiago},
  title     = {Métodos Avançados em Inferência Estatística Não-Paramétrica},
  year      = {2026},
  note      = {Capítulos 7 e 36}
}
```

---

**Autor:** Luiz Tiago Wilcke  
**Contato acadêmico:** wilckeluiztiago@gmail.com  
**Repositório original de modelos:** https://github.com/wilckeluiztiago-dotcom/Modelos_NaoParametricos
