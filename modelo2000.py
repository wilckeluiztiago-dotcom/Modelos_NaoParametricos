"""
PROJETO: REDES NEURAIS FÍSICO-INFORMADAS (PINNs) EM EQUAÇÕES DIFERENCIAIS ESTOCÁSTICAS
         PARA TRANSPORTE QUÂNTICO EM NANOTRANSISTORES DE 2 nm
Autor: Luiz Tiago Wilcke
Descrição: Script autocontido em bloco único contendo:
           1. Constantes físicas e parâmetros de escala para GAAFET de 2 nm.
           2. Arquitetura profunda da PINN (Poisson 1D + EDE de Itô / Fokker-Planck).
           3. Cálculo de perda informada pela física com Flutuação Estocástica de Dopantes (RDF).
           4. Integrador estocástico de Euler-Maruyama para trajetórias de partículas.
           5. Pipeline completo de treinamento, avaliação e plotagem dos 4 gráficos.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ==============================================================================
# 1. CONSTANTES FÍSICAS E PARÂMETROS DO CANAL (NANOSHEET 2 nm)
# ==============================================================================
# Fixação de sementes para reprodutibilidade
torch.manual_seed(42)
np.random.seed(42)

# Constantes Fundamentais
CARGA_ELEMENTAR = 1.60217663e-19       # Carga do elétron (C)
PERMISSIVIDADE_VACUO = 8.85418781e-12  # Permissividade do vácuo (F/m)
CONSTANTE_BOLTZMANN = 1.380649e-23     # Constante de Boltzmann (J/K)
TEMPERATURA_OPERACAO = 300.0           # Temperatura ambiente (K)

# Geometria e Material (Canal de Silício)
COMPRIMENTO_CANAL = 12.0e-9            # Comprimento de canal L = 12 nm
LARGURA_NANOSHEET = 5.0e-9             # Largura do Nanosheet (m)
ESPESSURA_NANOSHEET = 3.0e-9           # Espessura do Nanosheet (m)
CONSTANTE_DIELETRICA_SI = 11.7         # Permissividade relativa do Si
PERMISSIVIDADE_SILICIO = CONSTANTE_DIELETRICA_SI * PERMISSIVIDADE_VACUO

# Transporte Quântico e Eletrostática
MOBILIDADE_ELETRONS = 0.04             # Mobilidade efetiva no nó de 2 nm (m^2 / V.s)
TENSAO_FONTE = 0.0                     # Potencial na Fonte V_source (V)
TENSAO_DRENO = 0.7                     # Potencial no Dreno V_DD (V)
DENSIDADE_TOTAL_PORTADORES = 1.0e24    # Densidade volumétrica de portadores (m^-3)
JANELA_TEMPORAL_MAXIMA = 1.0e-12       # Janela temporal transiente (1 ps)

# Parâmetros Termodinâmicos e Difusivos
TENSAO_TERMICA = (CONSTANTE_BOLTZMANN * TEMPERATURA_OPERACAO) / CARGA_ELEMENTAR
COEFICIENTE_DIFUSAO_EINSTEIN = MOBILIDADE_ELETRONS * TENSAO_TERMICA
VOLATILIDADE_BROWNIANA_SIGMA = np.sqrt(2.0 * COEFICIENTE_DIFUSAO_EINSTEIN)

# Fatores de Escala Adimensionais para Estabilidade Numérica da Rede
ESCALA_ESPACO_X = COMPRIMENTO_CANAL
ESCALA_TEMPO_T = JANELA_TEMPORAL_MAXIMA
ESCALA_POTENCIAL_PHI = TENSAO_DRENO


# ==============================================================================
# 2. ARQUITETURA DA REDE NEURAL INFORMADA PELA FÍSICA (PINN)
# ==============================================================================
class RedeNeuralPINNTransporte(nn.Module):
    """
    Rede Neural Profunda com saídas desacopladas para:
    - Potencial eletrostático phi(x, t)
    - Densidade de probabilidade P(x, t) (não-negativa via Softplus)
    - Velocidade de deriva b0(x, t)
    """
    def __init__(self, dimensao_oculta: int = 64, num_camadas: int = 4):
        super(RedeNeuralPINNTransporte, self).__init__()
        
        camadas = [nn.Linear(2, dimensao_oculta), nn.Tanh()]
        for _ in range(num_camadas - 1):
            camadas.extend([nn.Linear(dimensao_oculta, dimensao_oculta), nn.Tanh()])
            
        self.espinha_dorsal = nn.Sequential(*camadas)
        self.cabecote_saida = nn.Linear(dimensao_oculta, 3)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Normalização das variáveis de entrada para a faixa [0, 1]
        x_norm = x / ESCALA_ESPACO_X
        t_norm = t / ESCALA_TEMPO_T
        entradas = torch.cat([x_norm, t_norm], dim=1)
        
        caracteristicas = self.espinha_dorsal(entradas)
        projecao = self.cabecote_saida(caracteristicas)
        
        # Desnormalização e ativações físicas
        potencial_phi = projecao[:, 0:1] * ESCALA_POTENCIAL_PHI
        densidade_probabilidade_P = torch.nn.functional.softplus(projecao[:, 1:2])
        
        velocidade_referencia = MOBILIDADE_ELETRONS * (ESCALA_POTENCIAL_PHI / ESCALA_ESPACO_X)
        velocidade_deriva_b0 = projecao[:, 2:3] * velocidade_referencia
        
        return potencial_phi, densidade_probabilidade_P, velocidade_deriva_b0


# ==============================================================================
# 3. MODELAGEM DE RDF E CÁLCULO DE PERDA FÍSICA
# ==============================================================================
def gerar_perfil_dopantes_rdf(x: torch.Tensor) -> torch.Tensor:
    """
    Modela armadilhas e dopantes discretos aleatórios no canal de 2 nm (RDF).
    """
    posicoes_dopantes = torch.tensor([2.0e-9, 6.0e-9, 9.5e-9], device=x.device)
    carga_dopante = 1.0e24  # m^-3
    campo_dopantes = torch.zeros_like(x)
    raio_atomico = 0.6e-9   # Raio de dispersão gaussiana (nm)
    
    for pos in posicoes_dopantes:
        campo_dopantes += carga_dopante * torch.exp(
            -((x - pos) ** 2) / (2.0 * (raio_atomico ** 2))
        )
    return campo_dopantes

def calcular_perda_pinn(
    modelo: nn.Module,
    x_colocacao: torch.Tensor,
    t_colocacao: torch.Tensor,
    peso_fisica: float = 1.0,
    peso_contorno: float = 20.0
):
    """
    Calcula a função de perda composta minimizando os resíduos de Poisson 1D,
    Fokker-Planck (EDE), acoplamento de deriva e condições de contorno de Dirichlet.
    """
    x_colocacao.requires_grad_(True)
    t_colocacao.requires_grad_(True)
    
    phi, P, b0 = modelo(x_colocacao, t_colocacao)
    
    # Derivadas automáticas para o Potencial Eletrostático
    grad_phi_x = torch.autograd.grad(
        phi, x_colocacao, torch.ones_like(phi), create_graph=True
    )[0]
    grad_phi_xx = torch.autograd.grad(
        grad_phi_x, x_colocacao, torch.ones_like(grad_phi_x), create_graph=True
    )[0]
    
    # Derivadas automáticas para a Densidade de Probabilidade P(x, t)
    grad_P_t = torch.autograd.grad(
        P, t_colocacao, torch.ones_like(P), create_graph=True
    )[0]
    grad_P_x = torch.autograd.grad(
        P, x_colocacao, torch.ones_like(P), create_graph=True
    )[0]
    grad_P_xx = torch.autograd.grad(
        grad_P_x, x_colocacao, torch.ones_like(grad_P_x), create_graph=True
    )[0]
    
    # Fluxo convectivo: d(b0 * P) / dx
    fluxo_convectivo = b0 * P
    grad_fluxo_x = torch.autograd.grad(
        fluxo_convectivo, x_colocacao, torch.ones_like(fluxo_convectivo), create_graph=True
    )[0]
    
    # 1. Resíduo de Fokker-Planck (EDE de Transporte)
    residuo_fokker_planck = (
        grad_P_t + grad_fluxo_x - COEFICIENTE_DIFUSAO_EINSTEIN * grad_P_xx
    )
    perda_sde = torch.mean(residuo_fokker_planck ** 2) * (ESCALA_TEMPO_T ** 2)
    
    # 2. Resíduo de Poisson 1D com Flutuação de Dopantes (RDF)
    densidade_eletrons = DENSIDADE_TOTAL_PORTADORES * P
    densidade_dopantes = gerar_perfil_dopantes_rdf(x_colocacao)
    residuo_poisson = grad_phi_xx + (CARGA_ELEMENTAR / PERMISSIVIDADE_SILICIO) * (
        densidade_dopantes - densidade_eletrons
    )
    fator_escala_poisson = (ESCALA_ESPACO_X ** 4) / (ESCALA_POTENCIAL_PHI ** 2)
    perda_poisson = torch.mean(residuo_poisson ** 2) * fator_escala_poisson
    
    # 3. Consistência da Deriva com o Campo Elétrico: b0 = mu_n * d(phi)/dx
    velocidade_campo = MOBILIDADE_ELETRONS * grad_phi_x
    escala_v = (MOBILIDADE_ELETRONS * ESCALA_POTENCIAL_PHI / ESCALA_ESPACO_X) ** 2
    perda_deriva = torch.mean((b0 - velocidade_campo) ** 2) / escala_v
    
    # 4. Condições de Contorno de Dirichlet
    num_pontos_bc = 64
    x_fonte = torch.zeros((num_pontos_bc, 1), device=x_colocacao.device)
    x_dreno = torch.full((num_pontos_bc, 1), COMPRIMENTO_CANAL, device=x_colocacao.device)
    t_aleatorio = torch.rand((num_pontos_bc, 1), device=x_colocacao.device) * JANELA_TEMPORAL_MAXIMA
    
    phi_fonte_pred, _, _ = modelo(x_fonte, t_aleatorio)
    phi_dreno_pred, _, _ = modelo(x_dreno, t_aleatorio)
    
    perda_contorno = (
        torch.mean((phi_fonte_pred - TENSAO_FONTE) ** 2) +
        torch.mean((phi_dreno_pred - TENSAO_DRENO) ** 2)
    )
    
    perda_total = (
        perda_sde +
        peso_fisica * (perda_poisson + perda_deriva) +
        peso_contorno * perda_contorno
    )
    
    return perda_total, perda_sde, perda_poisson, perda_deriva, perda_contorno


# ==============================================================================
# 4. SIMULADOR ESTOCÁSTICO DE EULER-MARUYAMA (EDE DE ITÔ)
# ==============================================================================
def simular_trajetorias_euler_maruyama(
    grade_posicoes_x: np.ndarray,
    campo_deriva_b0: np.ndarray,
    num_particulas: int = 6,
    num_passos: int = 150,
    passo_tempo_dt: float = 2.0e-15
) -> tuple:
    """
    Integração de trajetórias individuais de elétrons sob EDE dX_t = b0(X_t)dt + sigma dW_t.
    """
    vetor_tempo_ps = np.linspace(0, num_passos * passo_tempo_dt, num_passos) * 1e12
    matriz_trajetorias = np.zeros((num_particulas, num_passos))
    
    for p in range(num_particulas):
        posicao_atual = np.random.uniform(0.5e-9, 2.0e-9)
        matriz_trajetorias[p, 0] = posicao_atual
        
        for s in range(1, num_passos):
            velocidade_local = np.interp(posicao_atual, grade_posicoes_x, campo_deriva_b0)
            incremento_wiener = np.random.normal(0.0, np.sqrt(passo_tempo_dt))
            
            posicao_atual = (
                posicao_atual +
                velocidade_local * passo_tempo_dt +
                VOLATILIDADE_BROWNIANA_SIGMA * incremento_wiener
            )
            
            posicao_atual = np.clip(posicao_atual, 0.0, COMPRIMENTO_CANAL)
            matriz_trajetorias[p, s] = posicao_atual
            
    return vetor_tempo_ps, matriz_trajetorias


# ==============================================================================
# 5. PIPELINE PRINCIPAL: TREINAMENTO, VALIDAÇÃO E PLOTAGEM
# ==============================================================================
def main():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando simulação PINN em dispositivo: {dispositivo}")
    
    # 1. Instanciação e Otimizadores
    modelo = RedeNeuralPINNTransporte(dimensao_oculta=64, num_camadas=4).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=2.0e-3)
    escalonador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=800)
    
    num_epocas = 800
    tamanho_lote = 512
    
    print("\n--- INICIANDO TREINAMENTO DA PINN PARA TRANSPORTE QUÂNTICO (2 nm) ---")
    for epoca in range(1, num_epocas + 1):
        otimizador.zero_grad()
        
        x_amostrado = torch.rand((tamanho_lote, 1), device=dispositivo) * COMPRIMENTO_CANAL
        t_amostrado = torch.rand((tamanho_lote, 1), device=dispositivo) * JANELA_TEMPORAL_MAXIMA
        
        perda_total, p_sde, p_pois, p_der, p_bc = calcular_perda_pinn(
            modelo, x_amostrado, t_amostrado
        )
        
        perda_total.backward()
        otimizador.step()
        escalonador.step()
        
        if epoca % 200 == 0 or epoca == 1:
            print(
                f"Época [{epoca:04d}/{num_epocas}] | "
                f"Perda Total: {perda_total.item():.4e} | "
                f"Res. SDE: {p_sde.item():.4e} | "
                f"Res. Poisson: {p_pois.item():.4e} | "
                f"Res. Contorno: {p_bc.item():.4e}"
            )
            
    print("Treinamento da PINN concluído com sucesso.")
    
    # 2. Avaliação Espacial
    modelo.eval()
    pontos_x = np.linspace(0, COMPRIMENTO_CANAL, 300)
    tensor_x = torch.tensor(pontos_x, dtype=torch.float32, device=dispositivo).view(-1, 1)
    
    with torch.no_grad():
        phi_pred, P_meio, b0_pred = modelo(tensor_x, torch.full_like(tensor_x, 0.5 * JANELA_TEMPORAL_MAXIMA))
        _, P_t1, _ = modelo(tensor_x, torch.full_like(tensor_x, 0.10 * JANELA_TEMPORAL_MAXIMA))
        _, P_t2, _ = modelo(tensor_x, torch.full_like(tensor_x, 0.50 * JANELA_TEMPORAL_MAXIMA))
        _, P_t3, _ = modelo(tensor_x, torch.full_like(tensor_x, 0.90 * JANELA_TEMPORAL_MAXIMA))
        
    x_nm = pontos_x * 1e9
    phi_numpy = phi_pred.cpu().numpy().flatten()
    b0_numpy = b0_pred.cpu().numpy().flatten()
    banda_conducao_Ec = -phi_numpy  # eV
    
    # 3. Simulação Estocástica de Trajetórias de Partículas (Euler-Maruyama)
    tempo_traj_ps, matriz_trajetorias = simular_trajetorias_euler_maruyama(
        grade_posicoes_x=pontos_x,
        campo_deriva_b0=b0_numpy,
        num_particulas=6,
        num_passos=150,
        passo_tempo_dt=2.0e-15
    )
    
    # 4. Geração dos Gráficos (4 Painéis)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # Painel (a): Potencial Eletrostático e Banda de Condução
    ax1 = axs[0, 0]
    ax1.plot(x_nm, phi_numpy, color='#1f77b4', lw=2.5, label=r'Potencial $\hat{\phi}(x)$')
    ax1.set_xlabel('Posição no Canal $x$ (nm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Potencial Eletrostático (V)', color='#1f77b4', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_nm, banda_conducao_Ec, color='#d62728', lw=2.0, linestyle='--', label=r'Banda $E_c(x)$')
    ax1_twin.set_ylabel('Energia de Condução $E_c$ (eV)', color='#d62728', fontsize=11, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='#d62728')
    
    for d_nm, rotulo in zip([2.0, 6.0, 9.5], ['RDF #1', 'RDF #2', 'RDF #3']):
        ax1.axvline(d_nm, color='gray', linestyle=':', alpha=0.7)
        ax1.text(d_nm, 0.52, rotulo, rotation=90, fontsize=8, color='dimgray')
    ax1.set_title('(a) Eletrostática & Diagrama de Energia no Canal (2 nm)', fontsize=12, fontweight='bold')
    
    # Painel (b): Distribuição de Probabilidade Quântica P(x, t)
    ax2 = axs[0, 1]
    ax2.plot(x_nm, P_t1.cpu().numpy(), color='#2ca02c', lw=2.2, label=r'$t = 0{,}10\ \mathrm{ps}$')
    ax2.plot(x_nm, P_t2.cpu().numpy(), color='#ff7f0e', lw=2.2, label=r'$t = 0{,}50\ \mathrm{ps}$')
    ax2.plot(x_nm, P_t3.cpu().numpy(), color='#9467bd', lw=2.2, label=r'$t = 0{,}90\ \mathrm{ps}$')
    ax2.set_xlabel('Posição no Canal $x$ (nm)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Densidade de Probabilidade $P(x, t)$', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Distribuição Quântica sob Flutuação de Dopantes', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fontsize=10)
    
    # Painel (c): Trajetórias Estocásticas Individuais de Elétrons
    ax3 = axs[1, 0]
    cores_particulas = plt.cm.tab10(np.linspace(0, 1, 6))
    for p in range(6):
        ax3.plot(tempo_traj_ps, matriz_trajetorias[p] * 1e9, lw=1.6, color=cores_particulas[p], alpha=0.85, label=f'Elétron #{p+1}')
    ax3.axhline(12.0, color='crimson', linestyle='--', lw=2.0, label='Dreno ($x = 12\ \mathrm{nm}$)')
    ax3.set_xlabel('Tempo $t$ (ps)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Posição do Elétron $X_t$ (nm)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Trajetórias Estocásticas Individuais (EDE de Itô)', fontsize=12, fontweight='bold')
    ax3.legend(frameon=True, fontsize=9, loc='upper left', ncol=2)
    
    # Painel (d): Mapa Espaço-Temporal 2D
    ax4 = axs[1, 1]
    grade_x_mesh = np.linspace(0, COMPRIMENTO_CANAL, 100)
    grade_t_mesh = np.linspace(0, JANELA_TEMPORAL_MAXIMA, 100)
    X_mesh, T_mesh = np.meshgrid(grade_x_mesh, grade_t_mesh)
    
    X_plano = torch.tensor(X_mesh.flatten(), dtype=torch.float32, device=dispositivo).view(-1, 1)
    T_plano = torch.tensor(T_mesh.flatten(), dtype=torch.float32, device=dispositivo).view(-1, 1)
    
    with torch.no_grad():
        _, P_plano, _ = modelo(X_plano, T_plano)
        matriz_P_2d = P_plano.cpu().numpy().reshape(100, 100)
        
    contorno = ax4.contourf(X_mesh * 1e9, T_mesh * 1e12, matriz_P_2d, levels=50, cmap='magma')
    barra_cores = fig.colorbar(contorno, ax=ax4)
    barra_cores.set_label('Densidade de Carga $P(x, t)$', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Posição $x$ (nm)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Tempo $t$ (ps)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Mapa Espaço-Temporal do Transporte no Canal', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('transporte_quantico_pinn_2nm_completo.png', dpi=300)
    print("Gráficos salvos com sucesso em 'transporte_quantico_pinn_2nm_completo.png'.")
    plt.show()

if __name__ == "__main__":
    main()
