import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# MODELAGEM ESTATÍSTICA NÃO-PARAMÉTRICA DE UM LED DE FÓTON ÚNICO (SPS)
# As equações e estimadores estatísticos aplicados foram extraídos do livro:
# Wilcke, Luiz Tiago (2026). "Métodos Avançados em Inferência Estatística Não-Paramétrica".
# ==============================================================================


class SimuladorLEDPhotonUnico:
    def __init__(self, numero_trajetorias=70, tempo_final=1.0, n_passos=1000):
        self.N = numero_trajetorias
        self.T = tempo_final
        self.n = n_passos
        self.dt = self.T / self.n
        self.tempo = np.linspace(0, self.T, self.n + 1)

    def funcao_deriva_b0(self, x):
        """
        Função de deriva b_0(x) (dinâmica de injeção e recombinação de portadores).
        Referência teórica: Capítulo 6, Seção 6.2.
        """
        return -2.0 * x + np.sin(np.pi * x)

    def funcao_volatilidade_sigma(self, x):
        """
        Coeficiente de difusão/volatilidade local sigma(x).
        """
        return 0.4 * (1.0 + 0.2 * x**2)

    def simular_trajetorias_euler_maruyama(self):
        """
        Simula N cópias de trajetórias estocásticas via o Esquema de Euler-Maruyama.
        Referência teórica: Capítulo 6, Seção 6.3, Equação 6.9.
        """
        matriz_trajetorias = np.zeros((self.N, self.n + 1))
        x_inicial = 0.2
        matriz_trajetorias[:, 0] = x_inicial

        for i in range(self.N):
            for k in range(self.n):
                incremento_wiener = np.random.normal(0.0, np.sqrt(self.dt))
                x_atual = matriz_trajetorias[i, k]

                b_val = self.funcao_deriva_b0(x_atual)
                s_val = self.funcao_volatilidade_sigma(x_atual)

                matriz_trajetorias[i, k + 1] = (
                    x_atual + b_val * self.dt + s_val * incremento_wiener
                )

        return matriz_trajetorias


class EstimadorProjecaoDerivaLED:
    def __init__(self, grau_base_m=4):
        self.m = grau_base_m

    def base_ortogonal_legendre(self, x, j):
        """
        Sistemas ortonormais de Legendre para projeção.
        Referência teórica: Capítulo 3, Seção 3.5 e Capítulo 5, Seção 5.5.
        """
        xp = np.clip(x / 1.5, -1.0, 1.0)
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

    def ajustar_estimador_deriva(self, trajetorias, T):
        """
        Calcula o estimador de projeção de mínimos quadrados para a deriva de EDEs.
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
                    self.base_ortogonal_legendre(x_s, j) for j in range(self.m)
                ])

                matriz_psi += np.outer(phi_vals, phi_vals) * dt
                vetor_z += phi_vals * d_xi

        matriz_psi /= (N * T)
        vetor_z /= (N * T)

        theta_otimo = np.linalg.solve(matriz_psi + 1e-6 * np.eye(self.m), vetor_z)
        return theta_otimo

    def prever_deriva(self, x, theta_otimo):
        phi_vals = np.array([
            self.base_ortogonal_legendre(x, j) for j in range(self.m)
        ])
        return np.dot(theta_otimo, phi_vals)


# ==============================================================================
# EXECUÇÃO DO EXPERIMENTO COMPUTACIONAL E ANÁLISE GRÁFICA
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando simulação estocástica e inferência não-paramétrica para o LED de Fóton Único...")

    # 1. Configuração e simulação das trajetórias do LED
    experimento = SimuladorLEDPhotonUnico(
        numero_trajetorias=80, tempo_final=1.0, n_passos=1000
    )
    conjunto_trajetorias = experimento.simular_trajetorias_euler_maruyama()

    # 2. Ajuste do Estimador de Projeção de Mínimos Quadrados
    estimador_projecao = EstimadorProjecaoDerivaLED(grau_base_m=4)
    vetor_coeficientes = estimador_projecao.ajustar_estimador_deriva(
        conjunto_trajetorias, experimento.T
    )

    # 3. Geração de pontos para validação e comparação gráfica
    grade_x = np.linspace(-1.0, 1.0, 70)
    deriva_teorica_real = experimento.funcao_deriva_b0(grade_x)
    deriva_estimada_proj = np.array([
        estimador_projecao.prever_deriva(x, vetor_coeficientes) for x in grade_x
    ])

    # 4. Plotagem gráfica comparativa dos resultados experimentais
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        grade_x, deriva_teorica_real, "k-", linewidth=2.5,
        label="Deriva Teórica Real $b_0(x)$ (Dinâmica de Portadores)"
    )
    plt.plot(
        grade_x, deriva_estimada_proj, "r--", linewidth=2.2,
        label="Estimador de Projeção $\\hat{b}_m(x)$ (Capítulo 7)"
    )
    plt.title(
        "Inferência Não-Paramétrica de Deriva em EDEs para LED de Fóton Único (Wilcke, 2026)",
        fontsize=11, fontweight="bold"
    )
    plt.xlabel("Intensidade / População de Portadores [x]", fontsize=10)
    plt.ylabel("Função de Deriva Estimada [b0(x)]", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    print("Simulação e análise computacional do LED de fóton único executadas com sucesso!")
