"""
=============================================================================
AUTOR: Luiz Tiago Wilcke
PROJETO: Framework Avançado de Inferência Não-Paramétrica para Nanodispositivos
         (Aplicação: Sensor Quântico de Grafeno - GQD sob EDEs de Itô-Lévy)
DESCRIÇÃO: Implementação computacional robusta contendo:
           1. Simulação estocástica avançada (Capítulo 6 e 8)
           2. Projeção ortogonal de mínimos quadrados para deriva (Capítulos 3, 5 e 7)
           3. Estimação de volatilidade local via variação quadrática (Capítulo 18)
=============================================================================
"""
import numpy as np

# Configuração global de reprodutibilidade conforme padrão acadêmico
np.random.seed(2026)


class SimuladorGQDAvancado:
    """
    [BASE TEÓRICA: Capítulo 6 e Capítulo 8 - EDEs, Processos de Difusão e Saltos]
    Simula trajetórias estocásticas para o Sensor GQD baseadas em EDE de Itô-Lévy
    com volatilidade state-dependent e saltos de Poisson composto (Random Telegraph Signal).
    """
    def __init__(self, n_trajetorias=250, horizonte_T=3.0, n_passos=600):
        self.N = n_trajetorias
        self.T = horizonte_T
        self.n = n_passos
        self.dt = horizonte_T / n_passos

    def deriva_b0(self, x):
        """
        [BASE TEÓRICA: Seção 1.3 - Análise de Dados Funcionais e Deriva]
        Função de deriva teórica b_0(x) (Potencial eletrostático não-linear do GQD).
        """
        return -2.8 * np.tanh(x) - 1.2 * x

    def volatilidade_sigma0(self, x):
        """
        [BASE TEÓRICA: Capítulo 18.1 - O Problema de Estimação da Volatilidade Local]
        Função de volatilidade local sigma_0(x).
        """
        return 0.4 + 0.1 * x**2

    def simular_sistema(self, intensidade_poisson=0.08):
        """
        [BASE TEÓRICA: Capítulo 6.3 - Simulação Numérica via Esquema de Euler-Maruyama
         e Capítulo 8.1 - Dinâmicas Estocásticas com Saltos: Processos de Poisson Compostos]
        Gera a matriz de caminhos N x n utilizando discretização de Euler-Maruyama com saltos.
        """
        print(f"[*] Simulando {self.N} trajetórias de difusão de Itô-Lévy (Cap. 6 e 8)...")
        caminhos = np.zeros((self.N, self.n))
        caminhos[:, 0] = np.random.normal(0.0, 0.1, self.N)

        for k in range(1, self.n):
            # Incremento do Movimento Browniano padrão (dW_t ~ N(0, dt))
            dw = np.random.normal(0, np.sqrt(self.dt), self.N)

            # Processo de Poisson composto J_t para modelar saltos de armadilha (Cap. 8)
            checa_salto = np.random.binomial(1, intensidade_poisson * self.dt, self.N)
            tamanho_salto = checa_salto * np.random.normal(0.0, 0.25, self.N)

            x_ant = caminhos[:, k-1]
            b_val = self.deriva_b0(x_ant)
            sig_val = self.volatilidade_sigma0(x_ant)

            # Atualização discreta via Esquema de Euler-Maruyama (Cap. 6.3)
            caminhos[:, k] = x_ant + b_val * self.dt + sig_val * dw + tamanho_salto

        return caminhos


class ProjecaoMinimosQuadradosEDEs:
    """
    [BASE TEÓRICA: Capítulo 3.4 - O Estimador de Projeção de Mínimos Quadrados para Derivas,
     Capítulo 5 - Estimador de Mínimos Quadrados de Projeção Ortogonal, e
     Capítulo 7.3 - Definição do Estimador e Matriz de Design Contínua]
    """
    def __init__(self, caminhos, dt, ordem_m=4):
        self.caminhos = caminhos
        self.dt = dt
        self.N, self.n = caminhos.shape
        self.m = ordem_m
        self.incrementos_x = np.diff(caminhos, axis=1)

    def base_legendre_normalizada(self, x, j):
        """
        [BASE TEÓRICA: Capítulo 3.5 e 5.2 - Sistemas Ortonormais de Legendre]
        Calcula o j-ésimo polinômio de Legendre ortonormalizado no intervalo [-1, 1].
        """
        x_lim = np.clip(x, -2.0, 2.0) / 2.0
        if j == 0:
            return np.ones_like(x_lim) * np.sqrt(0.5)
        elif j == 1:
            return x_lim * np.sqrt(1.5)
        elif j == 2:
            return 0.5 * (3 * x_lim**2 - 1) * np.sqrt(2.5)
        elif j == 3:
            return 0.5 * (5 * x_lim**3 - 3 * x_lim) * np.sqrt(3.5)
        else:
            P_ant2, P_ant1 = np.ones_like(x_lim) * np.sqrt(0.5), x_lim * np.sqrt(1.5)
            P_atual = P_ant1
            for k in range(2, j + 1):
                P_atual = ((2 * k - 1) * x_lim * P_ant1 - (k - 1) * P_ant2) / k
                P_ant2, P_ant1 = P_ant1, P_atual
            return P_atual

    def ajustar_modelo_projecao(self):
        """
        [BASE TEÓRICA: Capítulo 7.3 - Matriz de Design Contínua e Vetor Z_m]
        Monta a matriz de design empírica e resolve o sistema linear de projeção ortogonal.
        """
        print(f"[*] Calculando matriz de design empírica e vetor Z_m (m={self.m}) (Cap. 7)...")
        T_total = self.n * self.dt
        Psi_m = np.zeros((self.m, self.m))
        Z_m = np.zeros(self.m)

        for i in range(self.N):
            for k in range(self.n - 1):
                ponto = self.caminhos[i, k]
                dk = self.incrementos_x[i, k]
                phi_vals = np.array([self.base_legendre_normalizada(np.array([ponto]), j)[0] for j in range(self.m)])

                # Acumulação empírica do produto interno contínuo (Cap. 7.3)
                Psi_m += np.outer(phi_vals, phi_vals) * self.dt
                Z_m += phi_vals * dk

        fator_escala = self.N * T_total
        Psi_m = Psi_m / fator_escala
        Z_m = Z_m / fator_escala

        # Regularização espectral de Tikhonov para garantir estabilidade numérica (Teorema 5.1 e 7.1)
        Psi_m += 1e-5 * np.eye(self.m)

        # Solução do sistema linear de mínimos quadrados de projeção (Equação 3.24 e 7.7)
        theta_hat = np.linalg.solve(Psi_m, Z_m)
        return theta_hat

    def avaliar(self, theta_hat, grade_x):
        """Reconstrói a função estimada no espaço."""
        res = np.zeros_like(grade_x)
        for idx, x in enumerate(grade_x):
            phi_vals = np.array([self.base_legendre_normalizada(np.array([x]), j)[0] for j in range(self.m)])
            res[idx] = np.dot(theta_hat, phi_vals)
        return res


class EstimadorVolatilidadeLocalEDEs:
    """
    [BASE TEÓRICA: Capítulo 18 - Estimação de Volatilidade Local via Variação Quadrática por Trajetórias Curtas]
    """
    def __init__(self, caminhos, dt):
        self.caminhos = caminhos
        self.dt = dt

    def estimar_volatilidade_local(self, grade_x, largura_h=0.35):
        """
        [BASE TEÓRICA: Capítulo 18.2 e 18.3 - Variação Quadrática Discreta e Estimador por Projeção]
        Aplica a regressão não-paramétrica implícita sobre os incrementos quadráticos.
        """
        print("[*] Estimando a Volatilidade Local via Variação Quadrática (Cap. 18)...")
        N, n = self.caminhos.shape
        inc = np.diff(self.caminhos, axis=1)
        var_quad = (inc**2) / self.dt  # Variação quadrática discreta local (Equação 18.3)

        vol_est = np.zeros_like(grade_x)
        for idx, x_alvo in enumerate(grade_x):
            num, den = 0.0, 0.0
            for i in range(N):
                for k in range(n - 1):
                    ponto = self.caminhos[i, k]
                    v = var_quad[i, k]
                    u = (ponto - x_alvo) / largura_h
                    peso = np.exp(-0.5 * u**2) / largura_h
                    den += peso
                    num += peso * v
            vol_est[idx] = num / den if den > 1e-8 else 0.0
        return vol_est


# =============================================================================
# EXECUTÁVEL PRINCIPAL DA APLICAÇÃO AVANÇADA COM FUNDAMENTAÇÃO TEÓRICA
# =============================================================================
if __name__ == "__main__":
    print("=====================================================================")
    print(" FRAMEWORK AVANÇADO: SENSOR QUÂNTICO GQD (INFERÊNCIA NÃO-PARAMÉTRICA)")
    print(f" Autor: Luiz Tiago Wilcke")
    print(f" Obra de Referência: Métodos Avançados em Inferência Estatística Não-Paramétrica")
    print("=====================================================================\n")

    # 1. Simulação Estocástica do Sistema (Capítulos 6 e 8)
    simulador = SimuladorGQDAvancado(n_trajetorias=300, horizonte_T=2.0, n_passos=400)
    caminhos_gqd = simulador.simular_sistema(intensidade_poisson=0.08)

    # 2. Configuração e Execução do Estimador de Deriva por Projeção de Legendre (Capítulos 3, 5 e 7)
    estimador_projecao = ProjecaoMinimosQuadradosEDEs(caminhos_gqd, simulador.dt, ordem_m=4)
    theta_otimo = estimador_projecao.ajustar_modelo_projecao()

    # 3. Avaliação dos Resultados da Deriva
    grade_espacial = np.linspace(-1.0, 1.0, 15)
    deriva_estimada = estimador_projecao.avaliar(theta_otimo, grade_espacial)
    deriva_teorica = simulador.deriva_b0(grade_espacial)

    print("\n--- RESULTADOS DA DERIVA ESTIMADA (Projeção Ortogonal - Cap. 7) ---")
    print(f"{'Posição (x)':<12} | {'Deriva Teórica b_0(x)':<22} | {'Deriva Estimada b_hat(x)':<24}")
    print("-" * 65)
    for i in range(0, len(grade_espacial), 3):
        print(f"{grade_espacial[i]:10.4f} | {deriva_teorica[i]:20.4f} | {deriva_estimada[i]:22.4f}")

    # 4. Estimação da Volatilidade Local (Capítulo 18)
    est_vol = EstimadorVolatilidadeLocalEDEs(caminhos_gqd, simulador.dt)
    vol_estimada = est_vol.estimar_volatilidade_local(grade_espacial, largura_h=0.4)
    vol_teorica = simulador.volatilidade_sigma0(grade_espacial)**2

    print("\n--- RESULTADOS DA VOLATILIDADE LOCAL (Cap. 18) ---")
    print(f"{'Posição (x)':<12} | {'Volatilidade Real a_0(x)':<24} | {'Volatilidade Estimada a_hat(x)':<26}")
    print("-" * 68)
    for i in range(0, len(grade_espacial), 3):
        print(f"{grade_espacial[i]:10.4f} | {vol_teorica[i]:22.4f} | {vol_estimada[i]:24.4f}")

    print("\n[+] Processamento analítico e computacional avançado concluído com êxito!")
