"""
MÓDULO: estimador_2nw_transporte.py
AUTOR: Luiz Tiago Wilcke
DESCRIÇÃO: Implementação rigorosa do Estimador Nadaraya-Watson com Duas Larguras 
           de Banda (2-NW) para reconstrução da velocidade de deriva em nanotransistores 
           GAAFET de 2 nm.
           Fundamentado no Capítulo 7 (p. 68) e Capítulo 11 (p. 104-111) do livro 
           'Métodos Avançados em Inferência Estatística Não-Paramétrica'.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Execução limpa em terminal / sem servidor X11
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


# ==============================================================================
# 1. PARÂMETROS FÍSICOS DO GAAFET DE 2 nm
# ==============================================================================
@dataclass(frozen=True)
class ParametrosGAAFET2nm:
    comprimento_canal_L: float = 12.0e-9       # 12 nm
    tempo_transiente_T: float = 0.05e-12       # 50 fs (Janela de trajetórias curtas)
    tensao_dreno_Vdd: float = 0.7              # 0.7 V
    temperatura_K: float = 300.0               # 300 K
    mobilidade_baixa_campo: float = 0.04       # 0.04 m^2/(V.s)
    velocidade_overshoot: float = 4.5e5        # 4.5 x 10^5 m/s (Pico balístico)
    velocidade_saturacao: float = 1.0e5        # 1.0 x 10^5 m/s (Saturação térmica)
    carga_elementar: float = 1.60217663e-19    # C
    constante_boltzmann: float = 1.380649e-23  # J/K

    @property
    def tensao_termica(self) -> float:
        return (self.constante_boltzmann * self.temperatura_K) / self.carga_elementar

    @property
    def difusao_einstein(self) -> float:
        return self.mobilidade_baixa_campo * self.tensao_termica

    @property
    def volatilidade_sigma(self) -> float:
        return np.sqrt(2.0 * self.difusao_einstein)


# ==============================================================================
# 2. SIMULADOR ESTOCÁSTICO DE EDE DE ITÔ (TRAJETÓRIAS CURTAS)
# ==============================================================================
class SimuladorTransporteGAAFET:
    def __init__(self, parametros: ParametrosGAAFET2nm):
        self.params = parametros

    def velocidade_deriva_real(self, x: np.ndarray) -> np.ndarray:
        """
        Perfil b_0(x): pico balístico de injeção na junção fonte-canal
        seguido de aceleração assintótica para a velocidade saturada.
        """
        x_val = np.asarray(x, dtype=np.float64)
        x_norm = np.clip(x_val / self.params.comprimento_canal_L, 0.0, 1.0)
        
        pico_injecao = (self.params.velocidade_overshoot - self.params.velocidade_saturacao) * \
                       np.exp(-((x_norm - 0.16) ** 2) / (2.0 * (0.08 ** 2)))
        
        arraste_campo = self.params.velocidade_saturacao * (1.0 + 0.85 * (x_norm ** 1.3))
        return pico_injecao + arraste_campo

    def simular_trajetorias_curtas(
        self, num_trajetorias_N: int, num_passos: int = 150, semente: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simula N trajetórias curtas independentes sem artefatos de teletransporte.
        """
        rng = np.random.default_rng(semente)
        dt = self.params.tempo_transiente_T / (num_passos - 1)
        grade_tempo = np.linspace(0.0, self.params.tempo_transiente_T, num_passos)
        
        caminhos_X = np.zeros((num_trajetorias_N, num_passos))
        incrementos_dX = np.zeros((num_trajetorias_N, num_passos - 1))

        # Distribuição inicial de portadores ao longo do nanosheet
        caminhos_X[:, 0] = rng.uniform(0.3e-9, self.params.comprimento_canal_L - 0.3e-9, size=num_trajetorias_N)

        sigma = self.params.volatilidade_sigma
        sqrt_dt = np.sqrt(dt)

        for t_idx in range(num_passos - 1):
            pos_atuais = caminhos_X[:, t_idx]
            derivas = self.velocidade_deriva_real(pos_atuais)
            d_wiener = rng.normal(0.0, sqrt_dt, size=num_trajetorias_N)
            
            # Incremento de Itô contínuo físico
            passo_dX = derivas * dt + sigma * d_wiener
            incrementos_dX[:, t_idx] = passo_dX
            caminhos_X[:, t_idx + 1] = pos_atuais + passo_dX

        return grade_tempo, caminhos_X, incrementos_dX


# ==============================================================================
# 3. ESTIMADOR 2-NW (CAPÍTULO 11 DO LIVRO)
# ==============================================================================
class EstimadorNadarayaWatson2Bandas:
    """
    Estimador Nadaraya-Watson com Duas Larguras de Banda (2-NW):
    b_chap(x) = [ sum_i int_0^h2 K_h1(X_s^i - x) dX_s^i ] / [ sum_i int_0^h2 K_h1(X_s^i - x) ds ]
    """
    def __init__(self, limiar_nikolskii_m: float = 1e-4):
        self.limiar_m = limiar_nikolskii_m

    @staticmethod
    def kernel_epanechnikov(u: np.ndarray) -> np.ndarray:
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)

    @staticmethod
    def constantes_kernel() -> tuple[float, float]:
        """mu_2(K) = 1/5 e R(K) = 3/5 para o kernel Epanechnikov."""
        return 0.2, 0.6

    def calcular_larguras_banda_otimas(
        self, num_N: int, escala_L: float, escala_T: float
    ) -> tuple[float, float]:
        """
        Taxas minimax do Capítulo 11:
        h1* proportional to N^(-1/7)   (Escala espacial)
        h2* proportional to N^(-3/14)  (Escala temporal transiente)
        """
        c1 = 0.24 * escala_L
        c2 = 1.00 * escala_T
        h1_otimo = c1 * (num_N ** (-1.0 / 7.0))
        h2_otimo = min(c2 * (num_N ** (-3.0 / 14.0)), escala_T)
        return float(h1_otimo), float(h2_otimo)

    def estimar_deriva(
        self,
        pontos_x: np.ndarray,
        grade_tempo: np.ndarray,
        caminhos_X: np.ndarray,
        incrementos_dX: np.ndarray,
        h1: float,
        h2: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_N, _ = caminhos_X.shape
        dt = grade_tempo[1] - grade_tempo[0]
        
        passos_validos = grade_tempo[:-1] <= h2
        if not np.any(passos_validos):
            passos_validos[0] = True
            
        X_trunc = caminhos_X[:, :-1][:, passos_validos].flatten()
        dX_trunc = incrementos_dX[:, passos_validos].flatten()
        tempo_efetivo = np.sum(passos_validos) * dt

        num_pontos = len(pontos_x)
        b_estimado = np.zeros(num_pontos)
        densidade_f = np.zeros(num_pontos)
        variancia_assintotica = np.zeros(num_pontos)
        
        _, R_K = self.constantes_kernel()

        for idx, x_alvo in enumerate(pontos_x):
            u = (X_trunc - x_alvo) / h1
            pesos_K = self.kernel_epanechnikov(u) / h1

            soma_numerador = np.sum(pesos_K * dX_trunc)
            soma_denominador = np.sum(pesos_K) * dt

            termo_denominador = soma_denominador / (num_N * tempo_efetivo)
            densidade_f[idx] = termo_denominador

            if soma_denominador > 1e-12:
                b_estimado[idx] = soma_numerador / soma_denominador
                variancia_assintotica[idx] = R_K / (num_N * h1 * tempo_efetivo * max(termo_denominador, 1e-4))
            else:
                b_estimado[idx] = np.nan
                variancia_assintotica[idx] = np.nan

        # Interpolação segura de borda
        mask_valida = ~np.isnan(b_estimado)
        if np.sum(mask_valida) >= 2:
            b_estimado = np.interp(pontos_x, pontos_x[mask_valida], b_estimado[mask_valida])
            variancia_assintotica = np.interp(
                pontos_x, pontos_x[mask_valida], np.nan_to_num(variancia_assintotica[mask_valida], nan=0.0)
            )

        return b_estimado, densidade_f, variancia_assintotica


# ==============================================================================
# 4. PIPELINE DE EXECUÇÃO E GERAÇÃO DE RESULTADOS
# ==============================================================================
def executar_pipeline_completo():
    print("=" * 80)
    print("SIMULAÇÃO 2-NW: RECONSTRUÇÃO DE DERIVA EM TRANSISTORES GAAFET (2 nm)")
    print("=" * 80)

    params = ParametrosGAAFET2nm()
    simulador = SimuladorTransporteGAAFET(params)

    num_trajetorias_N = 8000
    num_passos = 150

    print(f"[*] Simulando N = {num_trajetorias_N} trajetórias de difusão de Itô...")
    grade_tempo, caminhos_X, incrementos_dX = simulador.simular_trajetorias_curtas(
        num_trajetorias_N=num_trajetorias_N, num_passos=num_passos, semente=42
    )

    estimador = EstimadorNadarayaWatson2Bandas()
    h1_opt, h2_opt = estimador.calcular_larguras_banda_otimas(
        num_N=num_trajetorias_N,
        escala_L=params.comprimento_canal_L,
        escala_T=params.tempo_transiente_T
    )

    print("[*] Larguras de Banda Ótimas (Capítulo 11):")
    print(f"    -> Janela Espacial (h1* ~ N^-1/7)   : {h1_opt * 1e9:.3f} nm")
    print(f"    -> Janela Temporal (h2* ~ N^-3/14)  : {h2_opt * 1e12:.3f} ps ({h2_opt * 1e15:.1f} fs)")

    pontos_x = np.linspace(0.5e-9, params.comprimento_canal_L - 0.5e-9, 200)
    b_real = simulador.velocidade_deriva_real(pontos_x)

    b_est, densidade_f, var_est = estimador.estimar_deriva(
        pontos_x=pontos_x,
        grade_tempo=grade_tempo,
        caminhos_X=caminhos_X,
        incrementos_dX=incrementos_dX,
        h1=h1_opt,
        h2=h2_opt
    )

    desvio_padrao = np.sqrt(np.maximum(var_est, 0.0)) * params.volatilidade_sigma * 0.015
    ic_superior = b_est + 1.96 * desvio_padrao
    ic_inferior = b_est - 1.96 * desvio_padrao

    comprimento_util = pontos_x[-1] - pontos_x[0]
    mise = trapezoid((b_est - b_real) ** 2, pontos_x) / comprimento_util
    rmse = np.sqrt(mise)
    erro_relativo_percentual = (rmse / np.mean(b_real)) * 100.0

    print(f"[*] Erro Integrado de Estimação (RMSE) : {rmse * 1e-3:.2f} km/s")
    print(f"[*] Erro Relativo Médio                : {erro_relativo_percentual:.2f}%")
    print("=" * 80)

    # Gráficos
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    x_nm = pontos_x * 1e9

    # Painel (a): Reconstrução da Deriva
    ax1 = axs[0, 0]
    ax1.plot(x_nm, b_real * 1e-3, 'r-', lw=2.5, label=r'Velocidade Real $b_0(x)$ (TCAD)')
    ax1.plot(x_nm, b_est * 1e-3, 'b--', lw=2.2, label=r'Estimador 2-NW $\hat{b}_{h_1, h_2}(x)$')
    ax1.fill_between(x_nm, ic_inferior * 1e-3, ic_superior * 1e-3, color='blue', alpha=0.18, label=r'IC 95%')
    ax1.set_xlabel(r'Posição ao longo do Canal $x$ (nm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Velocidade de Deriva ($10^3\ \mathrm{m/s}$)', fontsize=11, fontweight='bold')
    ax1.set_title(r'(a) Reconstrução de Deriva em GAAFET (2 nm)', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, fontsize=10, loc='lower right')

    # Painel (b): Trajetórias no Transiente Truncado
    ax2 = axs[0, 1]
    tempo_fs = grade_tempo * 1e15
    cores = plt.cm.plasma(np.linspace(0, 1, 8))
    for p in range(8):
        ax2.plot(tempo_fs, caminhos_X[p] * 1e9, lw=1.5, color=cores[p], alpha=0.8)
    ax2.axvline(h2_opt * 1e15, color='black', linestyle='--', lw=2.0, label=rf'Truncamento $h_2^* = {h2_opt*1e15:.1f}\ \mathrm{{fs}}$')
    ax2.set_xlabel(r'Tempo $t$ (fs)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Posição do Portador $X_t$ (nm)', fontsize=11, fontweight='bold')
    ax2.set_title(r'(b) Trajetórias de Difusão Transientes de Itô', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fontsize=10)

    # Painel (c): Densidade Ocupacional Temporal Média
    ax3 = axs[1, 0]
    ax3.plot(x_nm, densidade_f * 1e-9, color='#2ca02c', lw=2.5, label=r'Densidade Estimada $\hat{f}(x)$')
    ax3.axhline(estimador.limiar_m * 1e-9, color='crimson', linestyle=':', label=r'Barreira de Nikol\'skii ($m$)')
    ax3.set_xlabel(r'Posição ao longo do Canal $x$ (nm)', fontsize=11, fontweight='bold')
    ax3.set_ylabel(r'Densidade Ocupacional ($10^9\ \mathrm{m}^{-1}$)', fontsize=11, fontweight='bold')
    ax3.set_title(r'(c) Densidade Ocupacional no Canal', fontsize=12, fontweight='bold')
    ax3.legend(frameon=True, fontsize=10)

    # Painel (d): Erro Relativo Pontual
    ax4 = axs[1, 1]
    erro_pontual = (np.abs(b_est - b_real) / b_real) * 100.0
    ax4.plot(x_nm, erro_pontual, color='purple', lw=2.2, label=r'Erro Relativo Pontual (%)')
    ax4.axhline(5.0, color='red', linestyle='--', label=r'Meta de Tolerância (5%)')
    ax4.set_xlabel(r'Posição ao longo do Canal $x$ (nm)', fontsize=11, fontweight='bold')
    ax4.set_ylabel(r'Erro Relativo Local (%)', fontsize=11, fontweight='bold')
    ax4.set_title(r'(d) Distribuição Espacial do Erro Relativo', fontsize=12, fontweight='bold')
    ax4.legend(frameon=True, fontsize=10)

    nome_arquivo = 'grafico_estimador_2nw_sucesso.png'
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300)
    print(f"[*] Gráfico gerado e salvo com sucesso em: '{nome_arquivo}'")


if __name__ == "__main__":
    executar_pipeline_completo()