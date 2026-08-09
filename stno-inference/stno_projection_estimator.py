import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, eigh
import time

# ==============================================================================
# MODELO COMPUTACIONAL: OSCILADOR NANOELETROMAGNÉTICO DE SPIN (STNO)
#
# AVISO DE ORIGEM TEÓRICA:
# Toda a teoria matemática subjacente, as Equações Diferenciais Estocásticas (EDEs),
# os estimadores de projeção, as matrizes de design empíricas, a estabilidade
# espectral e as U-estatísticas degeneradas implementadas neste código foram
# extraídas integralmente do livro:
# "Métodos Avançados em Inferência Estatística Não-Paramétrica" (Wilcke, 2026).
# ==============================================================================

class ParametrosFisicosSTNO:
    """
    Parâmetros físicos e spintrônicos do Oscilador Nanoeletromagnético de Spin.
    """
    def __init__(self):
        self.amortecimento_gilbert = 0.015
        self.razao_giremagnetica = 1.76e11
        self.constante_anisotropia = 4.5e4
        self.volume_camada = 5.0e-23
        self.temperatura_sistema = 300.0


class SimuladorEstocasticoEDEs:
    """
    Simulador de EDEs de Itô baseado no paradigma de múltiplas trajetórias curtas.

    ORIGEM DA TEORIA NO LIVRO:
    - Capítulo 6, Seção 6.2: Fundamentação das Equações Diferenciais Estocásticas.
    - Capítulo 6, Seção 6.3: Simulação Numérica via Esquema de Euler-Maruyama.
    - Capítulo 7, Seção 7.1: O Paradigma de Múltiplas Trajetórias Curtas Independentes.
    """
    def __init__(self, config_fisica, num_trajetorias=100, tempo_final=1.0, passos_n=1000):
        self.fisica = config_fisica
        self.numero_trajetorias = num_trajetorias
        self.tempo_final = tempo_final
        self.n_passos = passos_n
        self.passo_dt = self.tempo_final / self.n_passos
        self.malha_temporal = np.linspace(0, self.tempo_final, self.n_passos + 1)

    def funcao_deriva_drift_b0(self, x):
        """
        Função de deriva b_0(x) (torque de Słonczewski e amortecimento).
        Referência: Capítulo 6, Seção 6.2 do livro.
        """
        return -2.5 * np.sin(x) + 1.2 * np.sin(2.0 * x) - 0.6 * x

    def funcao_volatilidade_sigma0(self, x):
        """
        Função de difusão ou volatilidade local sigma_0(x) do sistema.
        Referência: Capítulo 6, Seção 6.2 do livro.
        """
        return 0.40 * (1.0 + 0.15 * np.cos(x)**2)

    def simular_trajetorias_euler_maruyama(self):
        """
        Executa a simulação discreta de Euler-Maruyama para N trajetórias estocásticas.
        Referência: Capítulo 6, Seção 6.3, Equação 6.9 do livro.
        """
        print(f"[Simulação EDE] Gerando {self.numero_trajetorias} trajetórias via Euler-Maruyama...")
        matriz_caminhos = np.zeros((self.numero_trajetorias, self.n_passos + 1))
        estado_inicial = 0.20
        matriz_caminhos[:, 0] = estado_inicial

        for i in range(self.numero_trajetorias):
            for k in range(self.n_passos):
                incremento_browniano = np.random.normal(0.0, np.sqrt(self.passo_dt))
                x_atual = matriz_caminhos[i, k]

                b_val = self.funcao_deriva_drift_b0(x_atual)
                s_val = self.funcao_volatilidade_sigma0(x_atual)

                matriz_caminhos[i, k + 1] = (
                    x_atual + b_val * self.passo_dt + s_val * incremento_browniano
                )

        print("[Simulação EDE] Trajetórias simuladas com sucesso.")
        return matriz_caminhos


class BaseOrthonormalLegendre:
    """
    Constrói o sistema ortonormal de Legendre para aproximação em espaços de Hilbert.

    ORIGEM DA TEORIA NO LIVRO:
    - Capítulo 3, Seção 3.5: Polinômios Ortogonais de Jacobi e Legendre.
    - Capítulo 5, Seção 5.2 e 5.5: Bases Ortonormais Clássicas e Sistemas de Legendre.
    """
    def __init__(self, grau_base_m=6):
        self.m = grau_base_m

    def avaliar_funcao_base(self, x, j):
        """
        Avalia o j-ésimo polinômio ortonormal de Legendre normalizado.
        """
        xp = np.clip(x / np.pi, -1.0, 1.0)
        if j == 0:
            return np.sqrt(0.5)
        elif j == 1:
            return np.sqrt(1.5) * xp
        elif j == 2:
            return np.sqrt(2.5) * 0.5 * (3.0 * xp**2 - 1.0)
        elif j == 3:
            return np.sqrt(3.5) * 0.5 * (5.0 * xp**3 - 3.0 * xp)
        elif j == 4:
            return np.sqrt(4.5) * 0.125 * (35.0 * xp**4 - 30.0 * xp**2 + 3.0)
        elif j == 5:
            return np.sqrt(5.5) * 0.125 * (63.0 * xp**5 - 70.0 * xp**3 + 15.0 * xp)
        else:
            return np.sqrt(j + 0.5) * (xp**j)

    def vetor_bases_completo(self, x):
        return np.array([self.avaliar_funcao_base(x, j) for j in range(self.m)])


class EstimadorProjecaoMinimosQuadrados:
    r"""
    Implementa o estimador por projeção de mínimos quadrados para derivas de EDEs.

    ORIGEM DA TEORIA NO LIVRO:
    - Capítulo 3, Seção 3.4.1: Função Objetivo Estocástica \gamma_N(b).
    - Capítulo 7, Seção 7.3: Definição do Estimador e Matriz de Design Contínua.
    - Capítulo 7, Seção 7.3, Equação 7.7: Resolução do sistema linear \hat{\Psi}_m \hat{\theta} = \hat{Z}_m.
    """
    def __init__(self, base_ortogonal):
        self.base = base_ortogonal
        self.m = base_ortogonal.m

    def computar_matrizes_design_empiricas(self, caminhos, tempo_final):
        r"""
        Calcula a Matriz de Design Empírica Contínua \hat{\Psi}_m e o Vetor Estocástico \hat{Z}_m.
        Referência: Capítulo 7, Seção 7.3, Equações 7.5 e 7.6 do livro.
        """
        print(r"[Inferência] Calculando matriz de design \hat{\Psi}_m e vetor estocástico \hat{Z}_m...")
        N, n_mais_um = caminhos.shape
        n_passos = n_mais_um - 1
        dt = tempo_final / n_passos

        matriz_psi_hat = np.zeros((self.m, self.m))
        vetor_z_hat = np.zeros(self.m)

        for i in range(N):
            for k in range(n_passos):
                x_s = caminhos[i, k]
                d_xi = caminhos[i, k + 1] - caminhos[i, k]
                phi_vals = self.base.vetor_bases_completo(x_s)

                matriz_psi_hat += np.outer(phi_vals, phi_vals) * dt
                vetor_z_hat += phi_vals * d_xi

        matriz_psi_hat /= (N * tempo_final)
        vetor_z_hat /= (N * tempo_final)

        return matriz_psi_hat, vetor_z_hat

    def ajustar_modelo_deriva(self, caminhos, tempo_final):
        r"""
        Resolve o sistema linear normal de projeção ortogonal \hat{\Psi}_m \hat{\theta} = \hat{Z}_m.
        Referência: Capítulo 7, Seção 7.3, Equação 7.7 do livro.
        """
        matriz_psi, vetor_z = self.computar_matrizes_design_empiricas(caminhos, tempo_final)

        autovalores = eigh(matriz_psi, driver="evd")[0]
        lambda_min = np.min(autovalores)
        lambda_max = np.max(autovalores)

        print(
            r"[Análise Espectral] Autovalores da matriz \hat{\Psi}_m -> "
            f"Min: {lambda_min:.6e} | Max: {lambda_max:.6e}"
        )

        fator_penalizacao = 1.0e-5 * lambda_max
        matriz_psi_estabilizada = matriz_psi + fator_penalizacao * np.eye(self.m)

        theta_otimo = solve(matriz_psi_estabilizada, vetor_z)
        return theta_otimo, lambda_min, matriz_psi

    def prever_funcao_deriva(self, x, theta_otimo):
        phi_vals = self.base.vetor_bases_completo(x)
        return np.dot(theta_otimo, phi_vals)


class AnalisadorUStatisticsDegeneradas:
    """
    Analisa flutuações de segunda ordem e resíduos estocásticos.

    ORIGEM DA TEORIA NO LIVRO:
    - Capítulo 17, Seção 17.2: A Formulação Contínua e a Decomposição de Hoeffding.
    - Capítulo 17, Seção 17.3: O Teorema do Limite Central para U-Estatísticas Degeneradas.
    """
    def __init__(self):
        pass

    def avaliar_u_estatistica_continua(self, caminhos):
        print("[U-Estatísticas] Calculando U-estatística degenerada contínua de 2ª ordem...")
        N = caminhos.shape[0]
        soma_acumulada = 0.0
        pares_contados = 0
        limite_amostra = min(N, 20)

        for i in range(limite_amostra):
            for j in range(i + 1, limite_amostra):
                diferenca = caminhos[i] - caminhos[j]
                nucleo_h2 = np.mean(diferenca**2) * np.exp(-0.5 * np.var(diferenca))
                soma_acumulada += nucleo_h2
                pares_contados += 1

        u_estatistica_valor = soma_acumulada / max(1, pares_contados)
        print(f"[U-Estatísticas] Valor final da U-estatística U_N(H_2): {u_estatistica_valor:.6e}")
        return u_estatistica_valor


class ValidadorRiscoIntegrado:
    """
    Avalia o risco integrado médio em L^2.
    Referência: Capítulo 3, Seção 3.4.4 e Capítulo 7, Seção 7.5 do livro.
    """
    @staticmethod
    def calcular_risco_l2(funcao_real, valores_estimados, grade):
        erros_quadraticos = (funcao_real(grade) - valores_estimados)**2
        # Substituído np.trapz por np.trapezoid para compatibilidade com NumPy 2.0+
        risco_l2 = np.trapezoid(erros_quadraticos, grade)
        return risco_l2


# ==============================================================================
# PIPELINE COMPUTACIONAL EXECUTIVO PRINCIPAL
# ==============================================================================

def executar_pipeline_completo_stno():
    tempo_inicio = time.time()

    print("========================================================================")
    print(" INICIANDO PIPELINE DE INFERÊNCIA ESTOCÁSTICA NÃO-PARAMÉTRICA")
    print(" Dispositivo: Oscilador Nanoeletromagnético de Spin (STNO)")
    print(" Teorias e Equações baseadas integralmente no Livro (Wilcke, 2026)")
    print("========================================================================")

    # 1. Instanciação e Simulação via EDE
    fisica = ParametrosFisicosSTNO()
    simulador = SimuladorEstocasticoEDEs(
        fisica, num_trajetorias=120, tempo_final=1.0, passos_n=1000
    )
    caminhos_amostrais = simulador.simular_trajetorias_euler_maruyama()

    # 2. Configuração da Base de Legendre e Estimação de Projeção
    base_legendre = BaseOrthonormalLegendre(grau_base_m=6)
    estimador = EstimadorProjecaoMinimosQuadrados(base_legendre)

    theta_otimo, lam_min, psi_matriz = estimador.ajustar_modelo_deriva(
        caminhos_amostrais, simulador.tempo_final
    )

    # 3. Avaliação de U-Estatísticas Degeneradas
    analisador_u = AnalisadorUStatisticsDegeneradas()
    u_val = analisador_u.avaliar_u_estatistica_continua(caminhos_amostrais)

    # 4. Validação de Risco Integrado L^2
    grade_espacial = np.linspace(-np.pi / 2, np.pi / 2, 80)
    deriva_real_vals = simulador.funcao_deriva_drift_b0(grade_espacial)
    deriva_est_vals = np.array([
        estimador.prever_funcao_deriva(x, theta_otimo) for x in grade_espacial
    ])

    risco_l2 = ValidadorRiscoIntegrado.calcular_risco_l2(
        simulador.funcao_deriva_drift_b0, deriva_est_vals, grade_espacial
    )
    print(f"[Validação] Risco Integrado L^2 Final: {risco_l2:.6e}")

    # 5. Visualização Gráfica Científica
    print("[Plotagem] Gerando gráfico comparativo analítico...")
    plt.figure(figsize=(11, 6))
    plt.plot(
        grade_espacial, deriva_real_vals, "k-", linewidth=3.0,
        label="Deriva Teórica Real $b_0(x)$ (STNO)"
    )
    plt.plot(
        grade_espacial, deriva_est_vals, "r--", linewidth=2.5,
        label="Estimador de Projeção Ortogonal $\\hat{b}_m(x)$ (Capítulo 7)"
    )
    plt.title(
        "Inferência Não-Paramétrica em EDEs para STNO via Projeção Ortogonal (Wilcke, 2026)",
        fontsize=12, fontweight="bold"
    )
    plt.xlabel("Ângulo de Precessão / Magnetização da Camada Livre [x]", fontsize=11)
    plt.ylabel("Função de Deriva Estimada [b0(x)]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    tempo_fim = time.time()
    print(f"[Execução Finalizada] Processo concluído com sucesso em {tempo_fim - tempo_inicio:.2f} segundos.")
    plt.show()


if __name__ == "__main__":
    executar_pipeline_completo_stno()
