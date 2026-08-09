"""
=============================================================================
AUTOR: Luiz Tiago Wilcke
PROJETO: Framework Estatístico Não-Paramétrico e Estocástico Completo
         para Nanodispositivos (Transistor de Átomo Único de Fósforo - Si:P)
DESCRIÇÃO: Implementação abrangente em Python contendo simulação avançada de EDE
           de Itô-Lévy com saltos, estimação não-paramétrica de deriva (Nadaraya-Watson),
           seleção adaptativa de largura de banda via PCO, estimação de volatilidade
           local e análise de confiabilidade (Função de Hazard).
=============================================================================
"""
import numpy as np
from scipy.stats import norm

# Configuração global de reprodutibilidade
np.random.seed(42)


class SimuladorEDELevyTransistor:
    """
    Classe responsável pela simulação estocástica avançada do transistor Si:P,
    incorporando deriva não-linear, difusão browniana e saltos de Poisson (RTS).
    Referência teórica: Capítulos 6 e 8.
    """
    def __init__(self, numero_trajetorias=300, horizonte_tempo=4.0, passos=600):
        self.numero_trajetorias = numero_trajetorias
        self.horizonte_tempo = horizonte_tempo
        self.passos = passos
        self.delta_t = horizonte_tempo / passos
        self.grade_tempos = np.linspace(0, horizonte_tempo, passos)

    def potencial_deriva_teorico(self, x):
        """
        Função de deriva b_0(x) real (Poço de potencial assimétrico do átomo Si:P).
        """
        return -2.5 * x * np.exp(-0.4 * x**2) + 0.3 * x

    def simular_trajetorias_completas(self, volatilidade_sigma=0.5, intensidade_poisson=0.1):
        """
        Simula N trajetórias independentes usando o Esquema de Euler-Maruyama
        estendido com saltos de Poisson composto.
        """
        print(f"[*] Simulando {self.numero_trajetorias} trajetórias estocásticas de Itô-Lévy...")
        matriz_trajetorias = np.zeros((self.numero_trajetorias, self.passos))

        # Condição inicial centrada no poço quântico do fósforo
        estado_inicial = np.random.normal(0.0, 0.15, self.numero_trajetorias)
        matriz_trajetorias[:, 0] = estado_inicial

        for k in range(1, self.passos):
            # Incrementos do Movimento Browniano padrão (Ruído Quântico/Térmico)
            incremento_wiener = np.random.normal(0, np.sqrt(self.delta_t), self.numero_trajetorias)

            # Processo de Poisson Composto para simular o Random Telegraph Signal (RTS)
            probabilidade_salto = intensidade_poisson * self.delta_t
            ocorrencia_salto = np.random.binomial(1, probabilidade_salto, self.numero_trajetorias)
            magnitude_salto = ocorrencia_salto * np.random.normal(0.0, 0.35, self.numero_trajetorias)

            # Passo de atualização numérica de Euler-Maruyama
            deriva_atual = self.potencial_deriva_teorico(matriz_trajetorias[:, k-1])
            matriz_trajetorias[:, k] = (
                matriz_trajetorias[:, k-1] +
                deriva_atual * self.delta_t +
                volatilidade_sigma * incremento_wiener +
                magnitude_salto
            )

        return matriz_trajetorias


class EstimadorNadarayaWatsonEDEs:
    """
    Implementa o Estimador de Nadaraya-Watson Contínuo para a função de deriva
    de EDEs a partir de múltiplas trajetórias curtas.
    Referência teórica: Capítulo 9.
    """
    def __init__(self, matriz_trajetorias, delta_t):
        self.matriz_trajetorias = matriz_trajetorias
        self.delta_t = delta_t
        self.n_trajetorias, self.n_passos = matriz_trajetorias.shape
        self.incrementos_x = np.diff(matriz_trajetorias, axis=1)  # dX_t

    def kernel_epanechnikov(self, u):
        """
        Função Kernel de Epanechnikov corrigida com np.where para suportar
        tanto valores escalares quanto arrays do NumPy sem erros de atribuição.
        """
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)

    def estimar_deriva(self, grade_x, largura_banda):
        """
        Calcula a estimativa não-paramétrica b_hat(x) para uma largura de banda h dada.
        """
        estimativa_deriva = np.zeros_like(grade_x)

        for idx, x_alvo in enumerate(grade_x):
            acumulador_numerador = 0.0
            acumulador_denominador = 0.0

            for i in range(self.n_trajetorias):
                for k in range(self.n_passos - 1):
                    ponto_amostral = self.matriz_trajetorias[i, k]
                    incremento_k = self.incrementos_x[i, k]

                    u = (ponto_amostral - x_alvo) / largura_banda
                    peso_kernel = self.kernel_epanechnikov(u) / largura_banda

                    acumulador_denominador += peso_kernel * self.delta_t
                    acumulador_numerador += peso_kernel * incremento_k

            if acumulador_denominador > 1e-8:
                estimativa_deriva[idx] = acumulador_numerador / acumulador_denominador
            else:
                estimativa_deriva[idx] = 0.0

        return estimativa_deriva


class SelecaoAdaptativaPCO:
    """
    Implementa o método PCO (Penalized Comparison to Overfitting) para a
    seleção automática e adaptativa da largura de banda h.
    Referência teórica: Capítulo 4.
    """
    def __init__(self, estimador_nw, grade_x, conjunto_larguras_banda):
        self.nw = estimador_nw
        self.grade_x = grade_x
        self.larguras_banda = conjunto_larguras_banda

    def selecionar_largura_banda_otima(self):
        """
        Varre o espaço de larguras de banda e aplica o critério de contraste penalizado PCO.
        """
        print("[*] Executando seleção adaptativa de largura de banda via PCO...")
        melhor_h = self.larguras_banda[0]
        menor_criterio = float('inf')

        # O menor h da coleção atua como a referência de sobreajuste (overfitting)
        h_min = min(self.larguras_banda)
        estimativa_referencia = self.nw.estimar_deriva(self.grade_x, h_min)

        resultados_pco = {}

        for h in self.larguras_banda:
            estimativa_atual = self.nw.estimar_deriva(self.grade_x, h)

            # Termo de contraste (distância L2 em relação ao sobreajuste de referência)
            contraste_R = np.sum((estimativa_atual - estimativa_referencia)**2)

            # Termo de penalização teórica PCO
            penalizacao_Pen = (2.0 / (self.nw.n_trajetorias * h)) - (0.01 / h)

            criterio_pco = contraste_R + penalizacao_Pen
            resultados_pco[h] = criterio_pco

            if criterio_pco < menor_criterio:
                menor_criterio = criterio_pco
                melhor_h = h

        print(f"[+] Largura de banda ótima selecionada via PCO: h_opt = {melhor_h:.4f}")
        return melhor_h, resultados_pco


class EstimadorVolatilidadeLocal:
    """
    Estima a função de volatilidade local quadrática a(x) = sigma^2(x)
    utilizando a variação quadrática discreta por trajetórias.
    Referência teórica: Capítulo 18.
    """
    def __init__(self, matriz_trajetorias, delta_t):
        self.matriz_trajetorias = matriz_trajetorias
        self.delta_t = delta_t

    def estimar_volatilidade(self, grade_x, largura_banda=0.4):
        print("[*] Estimando a Volatilidade Local via Variação Quadrática...")
        n_trajetorias, n_passos = self.matriz_trajetorias.shape
        incrementos_x = np.diff(self.matriz_trajetorias, axis=1)
        variacoes_quadraticas = (incrementos_x**2) / self.delta_t

        volatilidade_estimada = np.zeros_like(grade_x)

        for idx, x_alvo in enumerate(grade_x):
            numerador = 0.0
            denominador = 0.0

            for i in range(n_trajetorias):
                for k in range(n_passos - 1):
                    ponto = self.matriz_trajetorias[i, k]
                    v_quad = variacoes_quadraticas[i, k]

                    u = (ponto - x_alvo) / largura_banda
                    peso = np.exp(-0.5 * u**2) / largura_banda

                    denominador += peso
                    numerador += peso * v_quad

            if denominador > 1e-8:
                volatilidade_estimada[idx] = numerador / denominador
            else:
                volatilidade_estimada[idx] = 0.0

        return volatilidade_estimada


class AnaliseConfiabilidadeHazard:
    """
    Modela a confiabilidade e o risco de falha do transistor Si:P
    através do Estimador Suave da Função de Hazard.
    Referência teórica: Capítulo 15.
    """
    def __init__(self, tempos_de_vida):
        self.tempos_de_vida = tempos_de_vida
        self.n_amostras = len(tempos_de_vida)

    def estimar_hazard_suavizado(self, grade_tempos, largura_banda=0.3):
        print("[*] Calculando o Estimador Suave da Função de Hazard...")
        hazard_estimado = np.zeros_like(grade_tempos)

        for idx, t in enumerate(grade_tempos):
            densidade_acumulada = 0.0
            sobrevivencia_acumulada = 0.0

            for i in range(self.n_amostras):
                # Núcleo para a densidade
                u_dens = (t - self.tempos_de_vida[i]) / largura_banda
                densidade_acumulada += (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * u_dens**2) / largura_banda

                # Núcleo integrado para a fda
                u_fda = (t - self.tempos_de_vida[i]) / largura_banda
                fda_parcial = norm.cdf(u_fda)
                sobrevivencia_acumulada += (1.0 - fda_parcial)

            f_est = densidade_acumulada / self.n_amostras
            s_est = sobrevivencia_acumulada / self.n_amostras

            if s_est > 1e-4:
                hazard_estimado[idx] = f_est / s_est
            else:
                hazard_estimado[idx] = 0.0

        return hazard_estimado


# =============================================================================
# EXECUTÁVEL PRINCIPAL DO MODELO COMPLETO
# =============================================================================
if __name__ == "__main__":
    print("=====================================================================")
    print(" MODELO ESTATÍSTICO COMPLETO: TRANSISTOR DE FÓSFORO (Si:P)")
    print(" Autor: Luiz Tiago Wilcke")
    print("=====================================================================\n")

    # 1. Simulação Estocástica Avançada (250 trajetórias)
    simulador = SimuladorEDELevyTransistor(numero_trajetorias=250, horizonte_tempo=3.0, passos=400)
    matriz_caminhos = simulador.simular_trajetorias_completas(volatilidade_sigma=0.45, intensidade_poisson=0.12)

    # 2. Configuração do Estimador de Deriva por Nadaraya-Watson
    grade_espacial = np.linspace(-1.2, 1.2, 25)
    modulo_nw = EstimadorNadarayaWatsonEDEs(matriz_caminhos, simulador.delta_t)

    # 3. Seleção Adaptativa de Largura de Banda via PCO
    candidatos_h = [0.15, 0.25, 0.35, 0.50, 0.70]
    seletor_pco = SelecaoAdaptativaPCO(modulo_nw, grade_espacial, candidatos_h)
    h_otimo, dicionario_pco = seletor_pco.selecionar_largura_banda_otima()

    # 4. Estimação final da deriva com o h ótimo adaptativo
    deriva_estimada_otima = modulo_nw.estimar_deriva(grade_espacial, largura_banda=h_otimo)
    deriva_teorica = simulador.potencial_deriva_teorico(grade_espacial)

    print("\n--- RESULTADOS DA DERIVA ESTIMADA (Adaptativa PCO) ---")
    print(f"{'Posição (x)':<12} | {'Deriva Teórica b_0(x)':<22} | {'Deriva Estimada b_hat(x)':<24}")
    print("-" * 65)
    for i in range(0, len(grade_espacial), 5):
        print(f"{grade_espacial[i]:10.4f} | {deriva_teorica[i]:20.4f} | {deriva_estimada_otima[i]:22.4f}")

    # 5. Estimação de Volatilidade Local
    estimador_vol = EstimadorVolatilidadeLocal(matriz_caminhos, simulador.delta_t)
    volatilidade_estimada = estimador_vol.estimar_volatilidade(grade_espacial, largura_banda=0.4)

    print("\n--- RESULTADOS DA VOLATILIDADE LOCAL ---")
    print(f"{'Posição (x)':<12} | {'Volatilidade Local a(x) Estimada':<30}")
    print("-" * 45)
    for i in range(0, len(grade_espacial), 5):
        print(f"{grade_espacial[i]:10.4f} | {volatilidade_estimada[i]:28.4f}")

    # 6. Análise de Confiabilidade (Função de Hazard)
    tempos_falha_amostra = np.random.weibull(a=2.5, size=400) * 1.8
    grade_tempos_conf = np.linspace(0.1, 3.0, 20)
    modulo_hazard = AnaliseConfiabilidadeHazard(tempos_falha_amostra)
    hazard_suavizado = modulo_hazard.estimar_hazard_suavizado(grade_tempos_conf, largura_banda=0.35)

    print("\n--- RESULTADOS DE CONFIABILIDADE (Função de Hazard) ---")
    print(f"{'Tempo (t)':<12} | {'Taxa de Falha Estimada lambda(t)':<30}")
    print("-" * 48)
    for i in range(0, len(grade_tempos_conf), 5):
        print(f"{grade_tempos_conf[i]:10.4f} | {hazard_suavizado[i]:28.4f}")

    print("\n[+] Execução computacional do modelo estatístico encerrada com êxito!")
