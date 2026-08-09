#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIPELINE COMPUTACIONAL: NANODISPOSITIVO QUÂNTICO DE ALTA FREQUÊNCIA

AVISO DE ORIGEM TEÓRICA:
Toda a teoria matemática subjacente, as Equações Diferenciais Estocásticas,
os estimadores de projeção ortogonal, as matrizes de design empíricas contínuas,
a estabilidade espectral e as U-estatísticas degeneradas implementadas neste
código foram extraídas integralmente do livro:
"Métodos Avançados em Inferência Estatística Não-Paramétrica" (Wilcke, 2026).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, eigh
import time


class ParametrosFisicosNanodispositivo:
    """
    Parâmetros físicos e eletrostáticos do nanodispositivo quântico.
    Referência: Fundamentos de Confinamento Quântico e Transporte de Alta Frequência.
    """

    def __init__(self):
        self.capacitancia_canal = 1.0e-15      # [F] - Capacitância efetiva do canal
        self.frequencia_plasma = 1.2e12        # [Hz] - Frequência de plasma dos portadores
        self.temperatura_banho = 1.5           # [K] - Temperatura criogênica de operação
        self.constante_boltzmann = 1.380649e-23


class SimuladorEstocasticoNanodispositivo:
    """
    Simulador avançado de EDEs de Itô baseado no paradigma de múltiplas trajetórias curtas.

    ORIGEM DA TEORIA NO LIVRO:
    - Capítulo 6, Seção 6.2 e 6.3: Fundamentação e Simulação via Euler-Maruyama.
    - Capítulo 7, Seção 7.1: O Paradigma de Múltiplas Trajetórias Curtas Independentes.
    """

    def __init__(self, config_fisica, num_trajetorias=200, tempo_final=1.0, passos_n=1500):
        self.fisica = config_fisica
        self.N = num_trajetorias               # N cópias de trajetórias independentes (Capítulo 7.1)
        self.T = tempo_final                   # Horizonte temporal T fixo
        self.n = passos_n                      # Número de partições discretas n
        self.dt = self.T / self.n              # Passo temporal Δt = T/n
        self.malha_temporal = np.linspace(0, self.T, self.n + 1)

    def funcao_deriva_drift_b0(self, x):
        """
        Função de deriva b₀(x) (potencial eletrostático não-linear e confinamento).
        Referência teórica: Capítulo 6, Seção 6.2 do livro.
        """
        return -4.5 * x + 1.8 * x**3 - 0.7 * np.tanh(2.0 * x)

    def funcao_volatilidade_sigma0(self, x):
        """
        Coeficiente de difusão local σ₀(x) do nanodispositivo.
        Referência teórica: Capítulo 6, Seção 6.2 do livro.
        """
        return 0.32 * (1.0 + 0.18 * np.cos(x)**2)

    def simular_trajetorias_euler_maruyama(self):
        """
        Simula N cópias de trajetórias estocásticas via o Esquema de Euler-Maruyama.
        Referência teórica: Capítulo 6, Seção 6.3, Equação 6.9 do livro.
        """
        print(f"[Simulação EDE] Gerando {self.N} trajetórias via Euler-Maruyama (n={self.n})...")
        matriz_caminhos = np.zeros((self.N, self.n + 1))
        x_inicial = 0.0  # Estado eletrostático inicial x₀
        matriz_caminhos[:, 0] = x_inicial

        for i in range(self.N):
            for k in range(self.n):
                incremento_browniano = np.random.normal(0.0, np.sqrt(self.dt))
                x_atual = matriz_caminhos[i, k]

                b_val = self.funcao_deriva_drift_b0(x_atual)
                s_val = self.funcao_volatilidade_sigma0(x_atual)

                # EDE de Itô: X_{k+1} = X_k + b(X_k)Δt + σ(X_k)ΔW_k (Equação 6.9)
                matriz_caminhos[i, k + 1] = (
                    x_atual + b_val * self.dt + s_val * incremento_browniano
                )

        print("[Simulação EDE] Trajetórias simuladas com sucesso.")
        return matriz_caminhos


class BaseOrthonormalLegendreNano:
    """
    Constrói o sistema ortonormal de Legendre para aproximação em espaços de Hilbert.

    ORIGEM DA TEORIA NO LIVRO:
    - Capítulo 3, Seção 3.5: Polinômios Ortogonais de Jacobi e Legendre.
    - Capítulo 5, Seção 5.2 e 5.5: Bases Ortonormais Clássicas e Sistemas de Legendre.
    """

    def __init__(self, grau_base_m=6):
        self.m = grau_base_m  # Dimensão m do subespaço de aproximação S_m

    def avaliar_funcao_base(self, x, j):
        xp = np.clip(x / 1.5, -1.0, 1.0)
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


class EstimadorProjecaoMinimosQuadradosNano:
    """
    Implementa o estimador por projeção de mínimos quadrados para derivas de EDEs.

    ORIGEM DA TEORIA NO LIVRO:
    - Capítulo 3, Seção 3.4.1: Função Objetivo Estocástica γ_N(b).
    - Capítulo 7, Seção 7.3: Definição do Estimador e Matriz de Design Contínua Ψ̂_m.
    - Capítulo 7, Seção 7.3, Equação 7.7: Resolução do sistema linear Ψ̂_m θ̂ = Ẑ_m.
    """

    def __init__(self, base_ortogonal):
        self.base = base_ortogonal
        self.m = base_ortogonal.m

    def computar_matrizes_design_empiricas(self, caminhos, tempo_final):
        print("[Inferência] Calculando matriz de design empírica contínua Ψ̂_m e vetor Ẑ_m...")
        N, n_mais_um = caminhos.shape
        n_passos = n_mais_um - 1
        dt = tempo_final / n_passos
        matriz_psi_hat = np.zeros((self.m, self.m))
        vetor_z_hat = np.zeros(self.m)

        for i in range(N):
            for k in range(n_passos):
                x_s = caminhos[i, k]
                d_xi = caminhos[i, k + 1] - caminhos[i, k]  # Incremento estocástico dX_s
                phi_vals = self.base.vetor_bases_completo(x_s)

                # Produto interno empírico contínuo (Equação 7.5)
                matriz_psi_hat += np.outer(phi_vals, phi_vals) * dt
                # Vetor de dados estocástico via integral de Itô (Equação 7.6)
                vetor_z_hat += phi_vals * d_xi

        # Normalização por NT (Equações 7.5 e 7.6)
        matriz_psi_hat /= (N * tempo_final)
        vetor_z_hat /= (N * tempo_final)
        return matriz_psi_hat, vetor_z_hat

    def ajustar_modelo_deriva(self, caminhos, tempo_final):
        matriz_psi, vetor_z = self.computar_matrizes_design_empiricas(caminhos, tempo_final)

        # Análise espectral de estabilidade (Teorema 5.1 e Teorema 7.1)
        autovalores = eigh(matriz_psi, driver="evd")[0]
        lambda_min = np.min(autovalores)
        lambda_max = np.max(autovalores)
        print(
            f"[Análise Espectral] Autovalores da matriz empírica Ψ̂_m -> "
            f"Min: {lambda_min:.6e} | Max: {lambda_max:.6e}"
        )

        # Estabilização via Desigualdade de Bernstein Matricial (Capítulo 24.6, Teorema 24.2)
        fator_penalizacao = 1.0e-5 * lambda_max
        matriz_psi_estabilizada = matriz_psi + fator_penalizacao * np.eye(self.m)
        theta_otimo = solve(matriz_psi_estabilizada, vetor_z)
        return theta_otimo, lambda_min, matriz_psi

    def prever_funcao_deriva(self, x, theta_otimo):
        phi_vals = self.base.vetor_bases_completo(x)
        return np.dot(theta_otimo, phi_vals)


class AnalisadorUStatisticsDegeneradasNano:
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
        limite_amostra = min(N, 25)

        for i in range(limite_amostra):
            for j in range(i + 1, limite_amostra):
                diferenca = caminhos[i] - caminhos[j]
                # Núcleo simétrico H₂(Xⁱ, Xʲ) de segunda ordem baseado em Hoeffding
                nucleo_h2 = np.mean(diferenca**2) * np.exp(-0.25 * np.var(diferenca))
                soma_acumulada += nucleo_h2
                pares_contados += 1

        u_estatistica_valor = soma_acumulada / max(1, pares_contados)
        print(f"[U-Estatísticas] Valor final da U-estatística U_N(H₂): {u_estatistica_valor:.6e}")
        return u_estatistica_valor


class ValidadorRiscoIntegradoNano:
    """
    Avalia o risco integrado médio em L².
    Referência: Capítulo 3, Seção 3.4.4 e Capítulo 7, Seção 7.5 do livro.
    """

    @staticmethod
    def calcular_risco_l2(funcao_real, valores_estimados, grade):
        erros_quadraticos = (funcao_real(grade) - valores_estimados)**2
        # Utiliza np.trapezoid para plena compatibilidade com NumPy 2.0+
        risco_l2 = np.trapezoid(erros_quadraticos, grade)
        return risco_l2


# ==============================================================================
# PIPELINE COMPUTACIONAL EXECUTIVO PRINCIPAL
# ==============================================================================

def executar_pipeline_completo_nanodispositivo():
    tempo_inicio = time.time()
    print("========================================================================")
    print(" INICIANDO PIPELINE DE INFERÊNCIA ESTOCÁSTICA NÃO-PARAMÉTRICA")
    print(" Dispositivo: Nanodispositivo Quântico de Alta Frequência")
    print(" Teorias e Equações baseadas integralmente no Livro (Wilcke, 2026)")
    print("========================================================================")

    # 1. Simulação das trajetórias do nanodispositivo (Capítulo 6 e 7)
    fisica_nano = ParametrosFisicosNanodispositivo()
    simulador = SimuladorEstocasticoNanodispositivo(
        fisica_nano, num_trajetorias=200, tempo_final=1.0, passos_n=1500
    )
    caminhos_amostrais = simulador.simular_trajetorias_euler_maruyama()

    # 2. Configuração da Base de Legendre e Estimação de Projeção (Capítulos 3, 5 e 7)
    base_legendre = BaseOrthonormalLegendreNano(grau_base_m=6)
    estimador = EstimadorProjecaoMinimosQuadradosNano(base_legendre)
    theta_otimo, lam_min, psi_matriz = estimador.ajustar_modelo_deriva(
        caminhos_amostrais, simulador.T
    )

    # 3. Avaliação de U-Estatísticas Degeneradas (Capítulo 17)
    analisador_u = AnalisadorUStatisticsDegeneradasNano()
    u_val = analisador_u.avaliar_u_estatistica_continua(caminhos_amostrais)

    # 4. Validação de Risco Integrado L² (Capítulo 3 e 7)
    grade_espacial = np.linspace(-1.2, 1.2, 80)
    deriva_real_vals = simulador.funcao_deriva_drift_b0(grade_espacial)
    deriva_est_vals = np.array([
        estimador.prever_funcao_deriva(x, theta_otimo) for x in grade_espacial
    ])
    risco_l2 = ValidadorRiscoIntegradoNano.calcular_risco_l2(
        simulador.funcao_deriva_drift_b0, deriva_est_vals, grade_espacial
    )
    print(f"[Validação] Risco Integrado L² Final (Nanodispositivo): {risco_l2:.6e}")

    # 5. Visualização Gráfica Científica Avançada
    print("[Plotagem] Gerando gráfico comparativo analítico para o Nanodispositivo...")
    plt.figure(figsize=(11, 6))
    plt.plot(
        grade_espacial, deriva_real_vals, "k-", linewidth=3.0,
        label=r"Deriva Teórica Real $b_0(x)$ (Nanodispositivo Quântico)"
    )
    plt.plot(
        grade_espacial, deriva_est_vals, "r--", linewidth=2.5,
        label=r"Estimador de Projeção Ortogonal $\hat{b}_m(x)$ (Capítulo 7)"
    )
    plt.title(
        "Inferência Não-Paramétrica em EDEs para Nanodispositivo Quântico via Projeção (Wilcke, 2026)",
        fontsize=12, fontweight="bold"
    )
    plt.xlabel("Potencial Efetivo / Carga no Nanodispositivo [x]", fontsize=11)
    plt.ylabel("Função de Deriva Estimada [b₀(x)]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    tempo_fim = time.time()
    print(f"[Execução Finalizada] Processo concluído com sucesso em {tempo_fim - tempo_inicio:.2f} segundos.")
    plt.show()


if __name__ == "__main__":
    executar_pipeline_completo_nanodispositivo()
