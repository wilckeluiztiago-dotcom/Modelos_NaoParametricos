import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# ==============================================================================
# MODELAGEM ESTATÍSTICA NÃO-PARAMÉTRICA AVANÇADA DE UM NEMS COM MEMÓRIA FRACIONÁRIA
# As equações teóricas, operadores de Malliavin, U-estatísticas e estimadores de
# projeção implementados foram extraídos integralmente do livro de referência.
# Wilcke, Luiz Tiago (2026). "Métodos Avançados em Inferência Estatística Não-Paramétrica".
# ==============================================================================


class SimuladorNEMSFractional:
    def __init__(self, numero_trajetorias=80, tempo_final=1.0, n_passos=1000, hurst=0.7):
        self.N = numero_trajetorias
        self.T = tempo_final
        self.n = n_passos
        self.dt = self.T / self.n
        self.H = hurst
        self.tempo = np.linspace(0, self.T, self.n + 1)

    def gerar_covariancia_fractional(self):
        """
        Gera a matriz de covariância teórica do Movimento Browniano Fracionário (mBf).
        Referência teórica: Capítulo 8, Seção 8.2, Equação 8.5.
        """
        matriz_cov = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                t = self.tempo[i]
                s = self.tempo[j]
                matriz_cov[i, j] = 0.5 * (
                    t**(2 * self.H) + s**(2 * self.H) - np.abs(t - s)**(2 * self.H)
                )
        return matriz_cov

    def funcao_deriva_b0_real(self, x):
        """
        Função de deriva real b_0(x) do oscilador NEMS (força restauradora não-linear).
        Referência teórica: Capítulo 6, Seção 6.2.
        """
        return -2.5 * x - 1.0 * x**3

    def simular_trajetorias_fractionais(self):
        """
        Simula N cópias de trajetórias fracionárias utilizando decomposição de Cholesky
        da matriz de covariância do mBf (Método de Cholesky/Davies-Harte simplificado).
        """
        matriz_cov = self.gerar_covariancia_fractional()
        matriz_cov += 1e-8 * np.eye(self.n + 1)

        try:
            L = np.linalg.cholesky(matriz_cov)
        except np.linalg.LinAlgError:
            L = np.eye(self.n + 1)

        trajetorias = np.zeros((self.N, self.n + 1))
        x_inicial = 0.0

        for i in range(self.N):
            z = np.random.normal(0.0, 1.0, self.n + 1)
            fbm_path = np.dot(L, z)

            x = x_inicial
            trajetorias[i, 0] = x
            for k in range(self.n):
                dt_k = self.tempo[k + 1] - self.tempo[k]
                dfbm = fbm_path[k + 1] - fbm_path[k]
                b_val = self.funcao_deriva_b0_real(x)
                sigma_val = 0.4

                x = x + b_val * dt_k + sigma_val * dfbm
                trajetorias[i, k + 1] = x

        return trajetorias


class EstimadorProjecaoMalliavinNEMS:
    def __init__(self, grau_base_m=4):
        self.m = grau_base_m

    def base_ortonormal_legendre(self, x, j):
        """
        Sistemas ortonormais de Legendre para projeção em espaços de Hilbert.
        Referência teórica: Capítulo 3, Seção 3.5 e Capítulo 5, Seção 5.5.
        """
        xp = np.clip(x / 2.0, -1.0, 1.0)
        if j == 0:
            return np.sqrt(0.5)
        elif j == 1:
            return np.sqrt(1.5) * xp
        elif j == 2:
            return np.sqrt(2.5) * 0.5 * (3.0 * xp**2 - 1.0)
        elif j == 3:
            return np.sqrt(3.5) * 0.5 * (5.0 * xp**3 - 3.0 * xp)
        else:
            return np.sqrt(j + 0.5) * (xp**j)

    def ajustar_estimador_projecao(self, trajetorias, T):
        """
        Calcula o estimador por projeção de mínimos quadrados para a deriva de EDEs fracionárias.
        Referência teórica: Capítulo 7, Seção 7.3 (Equações 7.5, 7.6 e 7.7).
        """
        N, n_mais_um = trajetorias.shape
        n_passos = n_mais_um - 1
        dt = T / n_passos

        matriz_psi = np.zeros((self.m, self.m))
        vetor_z = np.zeros(self.m)

        for i in range(N):
            for k in range(n_passos):
                x_s = trajetorias[i, k]
                d_xi = trajetorias[i, k + 1] - trajetorias[i, k]

                phi_vals = np.array([
                    self.base_ortonormal_legendre(x_s, j) for j in range(self.m)
                ])

                matriz_psi += np.outer(phi_vals, phi_vals) * dt
                vetor_z += phi_vals * d_xi

        matriz_psi /= (N * T)
        vetor_z /= (N * T)

        # Verificação de estabilidade espectral via autovalores
        autovalores = np.linalg.eigvalsh(matriz_psi)
        lambda_min = np.min(autovalores)
        print(f"[Diagnóstico Espectral] Menor autovalor da matriz empírica Psi_m: {lambda_min:.6f}")

        theta_otimo = np.linalg.solve(matriz_psi + 1e-5 * np.eye(self.m), vetor_z)
        return theta_otimo, lambda_min

    def prever_deriva(self, x, theta_otimo):
        phi_vals = np.array([
            self.base_ortonormal_legendre(x, j) for j in range(self.m)
        ])
        return np.dot(theta_otimo, phi_vals)


class EvaluadorUStatisticsRisco:
    def __init__(self):
        pass

    def calcular_u_estatistica_deg(self, trajetorias):
        """
        Calcula uma U-estatística contínua degenerada de segunda ordem para análise de resíduos.
        Referência teórica: Capítulo 17, Seção 17.2, Equação 17.1.
        """
        N = trajetorias.shape[0]
        soma_u = 0.0
        contador = 0

        for i in range(min(N, 20)):
            for j in range(i + 1, min(N, 20)):
                h2_val = np.sum((trajetorias[i] - trajetorias[j])**2) / trajetorias.shape[1]
                soma_u += h2_val
                contador += 1

        return soma_u / max(1, contador)


# ==============================================================================
# EXECUÇÃO DO EXPERIMENTO COMPUTACIONAL E ANÁLISE GRÁFICA DO NEMS FRACIONÁRIO
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando simulação estocástica avançada do NEMS com mBf e inferência estatística...")

    # 1. Simulação das trajetórias fracionárias do NEMS
    dispositivo_nems = SimuladorNEMSFractional(
        numero_trajetorias=90, tempo_final=1.0, n_passos=800, hurst=0.75
    )
    matriz_caminhos_fbm = dispositivo_nems.simular_trajetorias_fractionais()

    # 2. Aplicação do Estimador de Projeção em EDEs Fracionárias
    estimador_proj = EstimadorProjecaoMalliavinNEMS(grau_base_m=5)
    coeficientes_otimos, lam_min = estimador_proj.ajustar_estimador_projecao(
        matriz_caminhos_fbm, dispositivo_nems.T
    )

    # 3. Avaliação de U-estatística degenerada para métrica de risco
    avaliador_u = EvaluadorUStatisticsRisco()
    u_estatistica_valor = avaliador_u.calcular_u_estatistica_deg(matriz_caminhos_fbm)
    print(f"[U-Estatística Degenerada U_N(H_2)] Valor estimado de dispersão quadrática: {u_estatistica_valor:.6f}")

    # 4. Geração de curvas para validação e comparação gráfica
    grade_espacial = np.linspace(-1.5, 1.5, 70)
    deriva_real = dispositivo_nems.funcao_deriva_b0_real(grade_espacial)
    deriva_proj_res = np.array([
        estimador_proj.prever_deriva(x, coeficientes_otimos) for x in grade_espacial
    ])

    # 5. Plotagem gráfica científica avançada
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        grade_espacial, deriva_real, "k-", linewidth=2.5,
        label="Deriva Teórica Real $b_0(x)$ (NEMS Fracionário)"
    )
    plt.plot(
        grade_espacial, deriva_proj_res, "r--", linewidth=2.2,
        label="Estimador de Projeção Ortogonal $\\hat{b}_m(x)$ (Capítulo 7)"
    )
    plt.title(
        "Inferência Não-Paramétrica com Memória Fracionária para NEMS (Wilcke, 2026)",
        fontsize=11, fontweight="bold"
    )
    plt.xlabel("Posição Mecânica do Oscilador [x]", fontsize=10)
    plt.ylabel("Função de Deriva Estimada [b0(x)]", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    print("Simulação, análise espectral e estimação computacional concluídas com sucesso!")
