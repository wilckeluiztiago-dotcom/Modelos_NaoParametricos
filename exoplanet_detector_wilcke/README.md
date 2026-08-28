# Detector de Exoplanetas por Trânsito Estelar – Métodos Não-Paramétricos

**Autor do modelo e implementação:** Luiz Tiago Wilcke  
**Base matemática:** *Métodos Avançados em Inferência Estatística Não-Paramétrica – Teoria Matemática, Processos de Difusão e Estimação por Kernel* (Luiz Tiago Wilcke, 2026)

## Visão Geral

Este projeto implementa um **pipeline completo de detecção de exoplanetas** a partir da **queda de iluminação** (trânsito) em curvas de luz de estrelas, utilizando **exclusivamente** a base teórica do livro.

O método clássico de trânsito detecta a diminuição periódica do fluxo estelar quando um planeta passa na frente da estrela. Aqui, em vez de modelos paramétricos rígidos (box-least-squares clássico, modelos de Mandel-Agol etc.), usamos:

- Estimadores de densidade e regressão por **kernel** (Rosenblatt-Parzen, Nadaraya-Watson)
- **Seleção adaptativa de largura de banda via PCO** (Penalized Comparison to Overfitting)
- **Estimadores de projeção de mínimos quadrados** ortogonais
- Modelagem do ruído estelar e variabilidade via **Equações Diferenciais Estocásticas (EDEs)** e processos de difusão
- Kernels **assimétricos** (Gama, Birnbaum-Saunders, Gaussiano Inverso) para bordas de trânsito
- Testes de aderência suavizados (Kolmogorov-Smirnov e Cramér-von Mises kernelizados)
- Bootstrap para processos estocásticos
- Classes de Nikol’skii e desigualdades de oráculo

### Partes do livro que inspiraram cada componente

| Módulo | Capítulo / Seção do Livro | Ideia Utilizada |
|--------|---------------------------|-----------------|
| 01–02  | Cap. 1 e 2                | Fundamentos de estimação não-paramétrica e espaços funcionais |
| 03     | Cap. 2 (Rosenblatt-Parzen)| Estimação de densidade do fluxo |
| 04     | Cap. 2 (Nadaraya-Watson)  | Suavização da curva de luz |
| 05     | Cap. 4 (Método PCO)       | Seleção adaptativa de banda |
| 06     | Cap. 3 (Redução de viés) | Kernels de ordem superior e Gama modificado |
| 07     | Cap. 5 e 7                | Projeção de mínimos quadrados para deriva e forma do trânsito |
| 08     | Cap. 9 e 11               | Nadaraya-Watson contínuo para detecção de dips |
| 09–10  | Cap. 6, 7, 8              | Modelagem de ruído por EDEs / difusões / saltos |
| 11     | Cap. 12–13                | Kernels assimétricos (Birnbaum-Saunders, Log-Normal, IG) |
| 12     | Cap. 15                   | Estimação da função de hazard (duração/intensidade do trânsito) |
| 13     | Cap. 19, 22               | Testes de aderência suavizados livres de fronteira |
| 14     | Cap. 23, 33               | Bootstrap paramétrico suavizado e de blocos |
| 15     | Cap. 20                   | Estimação de quantis suaves (Bahadur-Ghosh) |
| 16     | Cap. 21                   | Vida média residual (aplicada a fluxos residuals) |
| 17     | Cap. 32                   | Processos Gaussianos como prior não-paramétrico |
| 18     | Cap. 34                   | RKHS e teorema do representador |
| 19     | Cap. 35                   | Redes neurais como estimadores não-paramétricos |
| 20     | Cap. 29                   | Visualização e experimentos numéricos |
| 21     | Cap. 30 + integração      | Pipeline completo + perspectivas |

## Dados Reais da NASA

O pipeline baixa automaticamente curvas de luz reais do **Kepler** e **TESS** via a biblioteca `lightkurve` (arquivo MAST da NASA). Exemplos padrão:

- Kepler-10 (planeta confirmado)
- Kepler-8
- KIC 757450
- Qualquer TIC ou KIC disponível

## Estrutura do Projeto

```
exoplanet_detector_wilcke/
├── README.md
├── requirements.txt
├── main.py                 # Executa o pipeline completo
├── modulos/                # 21 módulos completos e interdependentes
│   ├── 01_carregamento_dados_nasa.py
│   ├── 02_pre_processamento.py
│   ├── 03_estimacao_densidade_kernel.py
│   ├── 04_regressao_nadaraya_watson.py
│   ├── 05_selecao_banda_pco.py
│   ├── 06_reducao_vies_kernel.py
│   ├── 07_estimador_projecao_minimos_quadrados.py
│   ├── 08_deteccao_transitos_kernel.py
│   ├── 09_modelagem_sde_ruido.py
│   ├── 10_estimacao_deriva_sde.py
│   ├── 11_kernels_assimetricos.py
│   ├── 12_estimacao_hazard_transito.py
│   ├── 13_testes_aderencia_suavizados.py
│   ├── 14_bootstrap_processos.py
│   ├── 15_estimacao_quantis_kernel.py
│   ├── 16_vida_media_residual.py
│   ├── 17_processos_gaussianos.py
│   ├── 18_rkhs_metodos_kernel.py
│   ├── 19_redes_neurais_nao_param.py
│   ├── 20_visualizacao_resultados.py
│   └── 21_pipeline_completo.py
├── dados/                  # Dados baixados / cache
├── resultados/             # Saídas (CSV, JSON, modelos)
└── figuras/                # Gráficos gerados
```

## Instalação e Execução

```bash
cd exoplanet_detector_wilcke
pip install -r requirements.txt

# Executar pipeline completo com estrela padrão (Kepler-10)
python main.py

# Ou especificar alvo
python main.py --alvo "Kepler-10" --missao Kepler
```

## Equações Centrais do Modelo 

### 1. Estimador de Densidade de Rosenblatt-Parzen (Cap. 2)

$$
\hat{f}_{X,h}(x) = \frac{1}{nh}\sum_{i=1}^{n} K\left(\frac{x-X_i}{h}\right)
$$

### 2. Regressão de Nadaraya-Watson (Cap. 2 e 9)

$$
\hat{m}_h(x) = \frac{\sum_{i=1}^{n} K\left(\frac{x-X_i}{h}\right) Y_i}{\sum_{i=1}^{n} K\left(\frac{x-X_i}{h}\right)}
$$

### 3. Critério PCO (Cap. 4)

$$
l_{\mathrm{PCO}}(h) = \left\|\hat{f}_{h_{\min}} - \hat{f}_h\right\|^2 - \frac{\left\|K_{h_{\min}} - K_h\right\|^2}{n} + \lambda \frac{\left\|K_h\right\|^2}{n}
$$

$$
\hat{h}_{\mathrm{PCO}} = \arg\min_{h \in \mathcal{H}} l_{\mathrm{PCO}}(h)
$$

### 4. Estimador de Projeção de Mínimos Quadrados (Cap. 5 e 7)

$$
\hat{b}_{m,N}(x) = \sum_{j=1}^{m} \hat{\theta}_j \varphi_j(x), \qquad \hat{\theta} = \hat{\Psi}_m^{-1} \hat{\Gamma}_m
$$

### 5. EDE para o ruído estelar (Cap. 6)

$$
dX_t = b(X_t)\,dt + \sigma(X_t)\,dW_t
$$

### 6. Kernel Gama de Chen / modificado (Cap. 3 e 12)

Usado para bordas de trânsito (fluxo $\geq 0$ após normalização). Forma típica do kernel Gama de Chen:

$$
K_{x,h}(t) = \frac{t^{x/h}\ e^{-t/h}}{h^{x/h+1}\ \Gamma(x/h+1)}, \quad t > 0
$$

## Autor

**Luiz Tiago Wilcke**  
Modelo construído integralmente a partir da teoria desenvolvida no livro *Métodos Avançados em Inferência Estatística Não-Paramétrica* (2026).

## Licença de Uso

Código livre para fins acadêmicos e de pesquisa, desde que citada a obra original de Luiz Tiago Wilcke.
