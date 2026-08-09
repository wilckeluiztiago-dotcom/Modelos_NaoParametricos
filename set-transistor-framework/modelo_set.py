import numpy as np
import matplotlib.pyplot as plt


class ModeloSETCompletoOtimizado:
    """
    Modelo Completo Otimizado de Inferência Não-Paramétrica para um SET (Single-Electron Transistor).
    As equações e algoritmos implementados abaixo derivam diretamente da teoria
    apresentada na obra de referência sobre inferência não-paramétrica em EDEs.
    """
    def __init__(self, n_trajetorias=1000, T=2.0, n_passos=1000):
        self.N = n_trajetorias
        self.T = T
        self.n = n_passos
        self.dt = T / n_passos
        self.m_basis = 5  # Dimensão do subespaço de aproximação m (Capítulos 5 e 7)

    def deriva_verdadeira(self, x):
        # Origem: Capítulo 6.2 - Equações Diferenciais Estocásticas e Difusões.
        # Representa a função de deriva (drift) b_0(x) de uma EDE de Itô.
        return -2.0 * np.sin(np.pi * x) - 1.0 * x

    def volatilidade_sigma(self, x):
        # Origem: Capítulo 6.2 e Capítulo 18.1 - Problema de Estimação da Volatilidade Local.
        # Modela a volatilidade local sigma_0(x) dependente do estado físico do dispositivo.
        return 0.4 * (1.0 + 0.5 * x**2)

    def simular_trajetorias_set(self):
        # Origem: Capítulo 6.3 - Simulação Numérica via Esquema de Euler-Maruyama.
        # Equação de discretização temporal recursiva de alta frequência.
        X = np.zeros((self.N, self.n))
        X[:, 0] = np.random.uniform(-0.5, 0.5, self.N)  # Condições iniciais
        for k in range(1, self.n):
            dW = np.sqrt(self.dt) * np.random.randn(self.N)
            x_ant = X[:, k-1]
            b = self.deriva_verdadeira(x_ant)
            sig = self.volatilidade_sigma(x_ant)
            X[:, k] = x_ant + b * self.dt + sig * dW
        return X

    def base_legendre(self, j, x):
        # Origem: Capítulo 3.5 e 5.2 - Polinômios Ortogonais de Legendre e Bases Ortonormais.
        # Define a base ortonormal normalizada no suporte espacial compactado.
        x_escalado = x / 1.5
        if j == 0:
            return 1.0 / np.sqrt(2.0)
        elif j == 1:
            return np.sqrt(3.0 / 2.0) * x_escalado
        elif j == 2:
            return np.sqrt(5.0 / 8.0) * (3.0 * x_escalado**2 - 1.0)
        elif j == 3:
            return np.sqrt(7.0 / 8.0) * (5.0 * x_escalado**3 - 3.0 * x_escalado)
        else:
            return np.sqrt(9.0 / 128.0) * (35.0 * x_escalado**4 - 30.0 * x_escalado**2 + 3.0)

    def ajustar_projecao_deriva(self, X):
        # Origem: Capítulo 7.3 - Definição do Estimador e a Matriz de Design Contínua.
        # Resolve o sistema de equações normais: \hat{\Psi}_m \hat{\theta} = \hat{Z}_m
        Psi = np.zeros((self.m_basis, self.m_basis))
        Z = np.zeros(self.m_basis)
        for i in range(self.N):
            dx = np.diff(X[i, :])
            for k in range(self.n - 1):
                x_val = X[i, k]
                phi = np.array([self.base_legendre(j, x_val) for j in range(self.m_basis)])
                Psi += np.outer(phi, phi) * self.dt
                Z += phi * dx[k]
        Psi = Psi / (self.N * self.T)
        Z = Z / (self.N * self.T)
        # Inversão regularizada da matriz gramiana empírica (Capítulo 5.4)
        theta = np.linalg.solve(Psi + 1e-4 * np.eye(self.m_basis), Z)
        return theta

    def ajustar_projecao_volatilidade_local(self, X):
        # Origem: Capítulo 18.3 - O Estimador por Projeção da Volatilidade Quadrática \hat{a}_m(x)
        # e Capítulo 18.2 - Variação Quadrática Discreta e a Identidade de Itô.
        Psi = np.zeros((self.m_basis, self.m_basis))
        Z_vol = np.zeros(self.m_basis)
        dx = np.diff(X, axis=1)
        for i in range(self.N):
            for k in range(self.n - 1):
                x_val = X[i, k]
                var_quad_local = (dx[i, k]**2) / self.dt  # Variação quadrática local instantânea
                phi = np.array([self.base_legendre(j, x_val) for j in range(self.m_basis)])
                Psi += np.outer(phi, phi) * self.dt
                Z_vol += phi * var_quad_local
        Psi = Psi / (self.N * self.T)
        Z_vol = Z_vol / (self.N * self.T)
        eta = np.linalg.solve(Psi + 1e-4 * np.eye(self.m_basis), Z_vol)
        return eta


# --- Execução Prática e Validação Numérica Otimizada ---
if __name__ == "__main__":
    print("Executando simulação otimizada com N=1000 trajetórias e alta frequência...")
    modelo = ModeloSETCompletoOtimizado(n_trajetorias=1000, T=2.0, n_passos=1000)
    dados_x = modelo.simular_trajetorias_set()

    # Aplicação dos estimadores não-paramétricos de alta ordem
    theta_deriva = modelo.ajustar_projecao_deriva(dados_x)
    eta_vol = modelo.ajustar_projecao_volatilidade_local(dados_x)

    # Geração da malha de avaliação analítica
    malha = np.linspace(-1.2, 1.2, 100)
    valores_deriva_real = modelo.deriva_verdadeira(malha)
    valores_deriva_estimada = np.array([
        sum(theta_deriva[j] * modelo.base_legendre(j, x) for j in range(modelo.m_basis))
        for x in malha
    ])
    valores_volatilidade_real = modelo.volatilidade_sigma(malha)**2  # Volatilidade quadrática teórica a_0(x)
    valores_volatilidade_estimada = np.array([
        sum(eta_vol[j] * modelo.base_legendre(j, x) for j in range(modelo.m_basis))
        for x in malha
    ])

    # Plotagem comparativa multi-painel
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5))

    eixos[0].plot(malha, valores_deriva_real, 'k--', linewidth=2.5, label='Deriva Teórica $b_0(x)$')
    eixos[0].plot(malha, valores_deriva_estimada, 'r-', linewidth=2.0, label='Projeção Ortogonal $\\hat{b}_m(x)$')
    eixos[0].set_title('Inferência Otimizada da Deriva em EDEs (Cap. 7)', fontsize=12)
    eixos[0].set_xlabel('Estado do Sistema ($x$)', fontsize=10)
    eixos[0].set_ylabel('Deriva Efetiva', fontsize=10)
    eixos[0].grid(True, linestyle=':', alpha=0.6)
    eixos[0].legend(frameon=True)

    eixos[1].plot(malha, valores_volatilidade_real, 'k--', linewidth=2.5, label='Volatilidade Real $a_0(x)$')
    eixos[1].plot(malha, valores_volatilidade_estimada, 'b-', linewidth=2.0, label='Variação Quadrática $\\hat{a}_m(x)$')
    eixos[1].set_title('Estimação Otimizada de Volatilidade Local (Cap. 18)', fontsize=12)
    eixos[1].set_xlabel('Estado do Sistema ($x$)', fontsize=10)
    eixos[1].set_ylabel('Variância Instantânea $\\sigma^2(x)$', fontsize=10)
    eixos[1].grid(True, linestyle=':', alpha=0.6)
    eixos[1].legend(frameon=True)

    plt.tight_layout()
    plt.show()
    print("Processo otimizado concluído com sucesso.")
