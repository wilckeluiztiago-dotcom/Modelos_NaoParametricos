#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
PINN-Itô para Transporte Quase-Balístico em GAAFETs Sub-2 nm
Autor: Luiz Tiago Wilcke
Física: Schrödinger 1D + Sub-bandas 2D + EDE de Itô para Quase-Fermi
===============================================================================
"""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Garante renderização headless para terminal e servidores
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# =============================================================================
# 1. CONSTANTES FÍSICAS E PARÂMETROS DO GAAFET SUB-2 NM
# =============================================================================
CARGA_ELEMENTAR = 1.60217663e-19            # q (C)
CONSTANTE_PLANCK = 6.62607015e-34           # h (J.s)
CONSTANTE_PLANCK_REDUZIDA = 1.0545718e-34   # \hbar (J.s)
MASSA_ELETRON_LIVRE = 9.10938356e-31        # m_0 (kg)
MASSA_EFETIVA_SI = 0.26 * MASSA_ELETRON_LIVRE  # m_eff = 0.26 m_0 (kg)
CONSTANTE_BOLTZMANN = 1.380649e-23          # k_B (J/K)
TEMPERATURA_K = 300.0                       # T = 300 K
TENSAO_TERMICA = (CONSTANTE_BOLTZMANN * TEMPERATURA_K) / CARGA_ELEMENTAR

# Dimensões do Canal Nanosheet GAAFET (Nó 2 nm)
COMPRIMENTO_CANAL_L = 10.0e-9               # L = 10 nm
LARGURA_NANOFIO_W = 2.5e-9                  # W = 2.5 nm
ALTURA_NANOFIO_H = 2.5e-9                   # H = 2.5 nm

# Energia de Corte da Sub-banda Fundamental E_{1,1}
ENERGIA_SUBBANDA_11 = (
    (CONSTANTE_PLANCK_REDUZIDA ** 2 * math.pi ** 2) / (2.0 * MASSA_EFETIVA_SI)
) * ((1.0 / LARGURA_NANOFIO_W**2) + (1.0 / ALTURA_NANOFIO_H**2)) / CARGA_ELEMENTAR  # em eV

# Tensões de Polarização Operacional
TENSAO_FONTE_VS = 0.00                      # V_S = 0.00 V
TENSAO_DRENO_VD = 0.65                      # V_D = 0.65 V
TENSAO_PORTA_VG = 0.70                      # V_G = 0.70 V

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. ARQUITETURA DA REDE NEURAL MULTITAREFA COM PROJEÇÃO DE FOURIER
# =============================================================================
class RedePINNItoGAAFET(nn.Module):
    """
    PINN-Itô Multitarefa:
      Entrada: (x, t) -> Posição longitudinal e Tempo
      Saídas: [\psi_R(x, t), \psi_I(x, t), \mu(x, t), b(x, t)]
    """
    def __init__(
        self,
        dimensao_entrada: int = 2,
        dimensao_saida: int = 4,
        neuronios_ocultos: int = 128,
        numero_camadas: int = 5
    ) -> None:
        super().__init__()
        
        # Mapeamento Espectral de Fourier
        self.dim_fourier = 64
        self.B_proj = nn.Parameter(
            torch.randn(dimensao_entrada, self.dim_fourier // 2) * 2.0,
            requires_grad=False
        )
        
        camadas = [
            nn.Linear(self.dim_fourier, neuronios_ocultos),
            nn.SiLU()
        ]
        for _ in range(numero_camadas - 1):
            camadas.append(nn.Linear(neuronios_ocultos, neuronios_ocultos))
            camadas.append(nn.SiLU())
            
        self.bloco_oculto = nn.Sequential(*camadas)
        self.camada_saida = nn.Linear(neuronios_ocultos, dimensao_saida)

    def forward(self, entradas: torch.Tensor) -> torch.Tensor:
        # Normalização espacial [0, L] e temporal [0, 1 ps]
        x_norm = entradas[:, 0:1] / COMPRIMENTO_CANAL_L
        t_norm = entradas[:, 1:2] / 1.0e-12
        entradas_norm = torch.cat([x_norm, t_norm], dim=-1)
        
        projecao = 2.0 * math.pi * torch.matmul(entradas_norm, self.B_proj)
        caracteristicas_fourier = torch.cat([torch.sin(projecao), torch.cos(projecao)], dim=-1)
        ativacao = self.bloco_oculto(caracteristicas_fourier)
        saidas = self.camada_saida(ativacao)
        return saidas


# =============================================================================
# 3. SIMULADOR DE TRAJETÓRIAS DE ITÔ E CONTRASTE DE MARIE
# =============================================================================
def simular_trajetorias_ito_portadores(
    numero_trajetorias: int = 128,
    passos_tempo: int = 100,
    tempo_total_T: float = 1.0e-12
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Simula N trajetórias curtas de difusão de Itô:
      d\mu_t = b_0(\mu_t) dt + \sigma dW_t
    """
    dt = tempo_total_T / passos_tempo
    grade_tempo = torch.linspace(0.0, tempo_total_T, passos_tempo + 1)
    
    volatilidade_sigma = math.sqrt(2.0 * CONSTANTE_BOLTZMANN * TEMPERATURA_K * 0.04 / CARGA_ELEMENTAR)
    trajetorias_mu = torch.zeros((numero_trajetorias, passos_tempo + 1))
    trajetorias_mu[:, 0] = TENSAO_FONTE_VS + torch.randn(numero_trajetorias) * (TENSAO_TERMICA * 0.2)

    for k in range(passos_tempo):
        mu_atual = trajetorias_mu[:, k]
        # Deriva não-linear de aceleração e saturação
        deriva_real = -6.0e12 * (mu_atual - TENSAO_DRENO_VD)
        difusao = volatilidade_sigma * math.sqrt(dt) * torch.randn(numero_trajetorias)
        trajetorias_mu[:, k + 1] = mu_atual + deriva_real * dt + difusao

    return grade_tempo, trajetorias_mu.to(DISPOSITIVO), dt


def calcular_contraste_ito_marie(
    modelo_pinn: nn.Module,
    trajetorias_mu: torch.Tensor,
    dt: float,
    tempo_total: float
) -> torch.Tensor:
    """
    Calcula \gamma_N(b) = (1/NT) \sum_{i=1}^N [ \int b^2 dt - 2 \int b d\mu ].
    """
    N_copias, n_amostras = trajetorias_mu.shape
    passos = n_amostras - 1
    
    mu_vetor = trajetorias_mu[:, :-1].reshape(-1, 1)
    t_vetor = torch.linspace(0.0, tempo_total, passos).repeat(N_copias).reshape(-1, 1).to(DISPOSITIVO)
    x_meio = torch.full_like(mu_vetor, COMPRIMENTO_CANAL_L * 0.5)
    
    entradas = torch.cat([x_meio, t_vetor], dim=1)
    saidas = modelo_pinn(entradas)
    b_pred = saidas[:, 3:4].reshape(N_copias, passos)
    
    # 1. Integral do quadrado da deriva: \int_0^T b^2 dt
    integral_b2 = torch.sum(b_pred ** 2, dim=1) * dt
    
    # 2. Integral estocástica de Itô: \int_0^T b d\mu
    incrementos_dmu = trajetorias_mu[:, 1:] - trajetorias_mu[:, :-1]
    integral_ito = torch.sum(b_pred * incrementos_dmu, dim=1)
    
    contraste_total = (1.0 / (N_copias * tempo_total)) * torch.sum(integral_b2 - 2.0 * integral_ito)
    return contraste_total


# =============================================================================
# 4. MOTOR DE TREINAMENTO DA PINN-ITÔ
# =============================================================================
def treinar_pinn_ito_completa(epocas: int = 400) -> nn.Module:
    modelo = RedePINNItoGAAFET().to(DISPOSITIVO)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-5)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas, eta_min=1e-5)
    
    tempo_T = 1.0e-12
    _, trajetorias_mu, dt_sim = simular_trajetorias_ito_portadores(
        numero_trajetorias=128, passos_tempo=100, tempo_total_T=tempo_T
    )

    print("=" * 85)
    print("TREINAMENTO DA PINN-ITÔ PARA TRANSPORTE QUÂNTICO BALÍSTICO (GAAFET 2 nm)")
    print(f"Autor: Luiz Tiago Wilcke | Dispositivo: {DISPOSITIVO}")
    print("=" * 85)

    for ep in range(1, epocas + 1):
        otimizador.zero_grad()
        
        # 1. Pontos de Colocação no Interior do Domínio
        N_col = 2048
        x_col = torch.rand(N_col, 1, requires_grad=True).to(DISPOSITIVO) * COMPRIMENTO_CANAL_L
        t_col = torch.rand(N_col, 1, requires_grad=True).to(DISPOSITIVO) * tempo_T
        entradas_col = torch.cat([x_col, t_col], dim=1)
        
        preds = modelo(entradas_col)
        psi_R = preds[:, 0:1]
        psi_I = preds[:, 1:2]
        mu_pot = preds[:, 2:3]
        
        # Derivadas Temporais de Primeira Ordem
        dpsiR_dt = torch.autograd.grad(psi_R, entradas_col, grad_outputs=torch.ones_like(psi_R), create_graph=True, retain_graph=True)[0][:, 1:2]
        dpsiI_dt = torch.autograd.grad(psi_I, entradas_col, grad_outputs=torch.ones_like(psi_I), create_graph=True, retain_graph=True)[0][:, 1:2]
        
        # Derivadas Espaciais de Segunda Ordem
        dpsiR_dx = torch.autograd.grad(psi_R, entradas_col, grad_outputs=torch.ones_like(psi_R), create_graph=True, retain_graph=True)[0][:, 0:1]
        dpsiI_dx = torch.autograd.grad(psi_I, entradas_col, grad_outputs=torch.ones_like(psi_I), create_graph=True, retain_graph=True)[0][:, 0:1]
        
        d2psiR_dx2 = torch.autograd.grad(dpsiR_dx, entradas_col, grad_outputs=torch.ones_like(dpsiR_dx), create_graph=True, retain_graph=True)[0][:, 0:1]
        d2psiI_dx2 = torch.autograd.grad(dpsiI_dx, entradas_col, grad_outputs=torch.ones_like(dpsiI_dx), create_graph=True, retain_graph=True)[0][:, 0:1]
        
        # Perfil de Barreira Eletrostática de Porta GAA 3D
        v_gate_perfil = TENSAO_PORTA_VG * torch.sin(math.pi * x_col / COMPRIMENTO_CANAL_L)
        potencial_efetivo = (v_gate_perfil + ENERGIA_SUBBANDA_11 - mu_pot)
        
        # Resíduos da Equação de Schrödinger 1D Decomposta
        res_real = (CONSTANTE_PLANCK_REDUZIDA * dpsiR_dt) - (
            -((CONSTANTE_PLANCK_REDUZIDA ** 2) / (2.0 * MASSA_EFETIVA_SI * CARGA_ELEMENTAR)) * d2psiI_dx2 + potencial_efetivo * psi_I
        ) * 1e12
        
        res_imag = (-CONSTANTE_PLANCK_REDUZIDA * dpsiI_dt) - (
            -((CONSTANTE_PLANCK_REDUZIDA ** 2) / (2.0 * MASSA_EFETIVA_SI * CARGA_ELEMENTAR)) * d2psiR_dx2 + potencial_efetivo * psi_R
        ) * 1e12
        
        perda_schrodinger = torch.mean(res_real ** 2 + res_imag ** 2) * 1e-24

        # 2. Contraste Estocástico de Itô
        perda_ito = calcular_contraste_ito_marie(modelo, trajetorias_mu, dt_sim, tempo_T) * 1e-24

        # 3. Condições de Contorno de Borda (Fonte e Dreno)
        N_bc = 256
        t_bc = torch.rand(N_bc, 1).to(DISPOSITIVO) * tempo_T
        
        bc_s = torch.cat([torch.zeros(N_bc, 1).to(DISPOSITIVO), t_bc], dim=1)
        bc_d = torch.cat([torch.full((N_bc, 1), COMPRIMENTO_CANAL_L).to(DISPOSITIVO), t_bc], dim=1)
        
        preds_s = modelo(bc_s)
        preds_d = modelo(bc_d)
        
        perda_bc_mu = torch.mean((preds_s[:, 2:3] - TENSAO_FONTE_VS) ** 2) + torch.mean((preds_d[:, 2:3] - TENSAO_DRENO_VD) ** 2)
        perda_bc_dens = torch.mean((preds_s[:, 0:1] ** 2 + preds_s[:, 1:2] ** 2 - 1.0) ** 2)

        perda_total = perda_schrodinger + 0.1 * perda_ito + 10.0 * (perda_bc_mu + perda_bc_dens)
        
        perda_total.backward()
        otimizador.step()
        agendador.step()

        if ep % 50 == 0 or ep == 1:
            print(
                f"Época {ep:3d}/{epocas} | "
                f"Perda Total: {perda_total.item():.5e} | "
                f"Schrödinger: {perda_schrodinger.item():.4e} | "
                f"Itô Contr.: {perda_ito.item():.4e}"
            )

    print("=" * 85)
    print("TREINAMENTO DA PINN-ITÔ CONCLUÍDO COM SUCESSO.")
    print("=" * 85)
    return modelo


# =============================================================================
# 5. EXTRAÇÃO DE PARÂMETROS BALÍSTICOS E GERAÇÃO DE GRÁFICOS
# =============================================================================
def avaliar_e_salvar_resultados(modelo: nn.Module) -> None:
    modelo.eval()
    
    n_pts = 100
    x_eval = np.linspace(0.0, COMPRIMENTO_CANAL_L, n_pts)
    t_eval = np.full(n_pts, 0.5e-12)
    
    entradas_eval = torch.tensor(np.column_stack([x_eval, t_eval]), dtype=torch.float32).to(DISPOSITIVO)
    
    with torch.no_grad():
        preds = modelo(entradas_eval).cpu().numpy()
        
    psi_R = preds[:, 0]
    psi_I = preds[:, 1]
    quase_fermi = preds[:, 2]
    densidade_prob = psi_R ** 2 + psi_I ** 2
    
    # Velocidade Balística de Injeção: v_drift = \sqrt{2 q \mu / m_eff}
    vel_balistica = np.sqrt(2.0 * CARGA_ELEMENTAR * np.maximum(quase_fermi, 1e-4) / MASSA_EFETIVA_SI)
    
    # Coeficiente de Transmissão Quântica de Landauer-Büttiker
    potencial_barreira = TENSAO_PORTA_VG * np.sin(np.pi * x_eval / COMPRIMENTO_CANAL_L) + ENERGIA_SUBBANDA_11
    energia_cinetica = quase_fermi
    transmissao = np.exp(-2.0 * np.sqrt(np.maximum(potencial_barreira - energia_cinetica, 0.0) * CARGA_ELEMENTAR * 2.0 * MASSA_EFETIVA_SI) * (COMPRIMENTO_CANAL_L / n_pts) / CONSTANTE_PLANCK_REDUZIDA)

    print("\n" + "=" * 80)
    print(f"{'Posição x (nm)':>14} | {'Quase-Fermi (V)':>16} | {'Velocidade (10^7 cm/s)':>24} | {'Transmissão Landauer':>20}")
    print("-" * 80)
    for i in range(0, n_pts, 10):
        print(f"{x_eval[i]*1e9:14.2f} | {quase_fermi[i]:16.4f} | {vel_balistica[i]*1e-5:24.4f} | {transmissao[i]:20.4f}")
    print("=" * 80)

    # Plotagem Científica e Salvamento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Quase-Nível de Fermi vs Barreira Quântica
    ax1.plot(x_eval * 1e9, quase_fermi, "b-", lw=2.2, label=r"Quase-Nível de Fermi $\mu_\theta(x)$")
    ax1.plot(x_eval * 1e9, potencial_barreira, "r--", lw=1.8, label=r"Barreira Quântica $E_{1,1} + qV_G(x)$")
    ax1.plot(x_eval * 1e9, densidade_prob * 0.1, "k:", lw=1.5, label=r"Densidade $|\psi(x)|^2$ (U.A.)")
    ax1.set_xlabel("Posição no Canal $x$ (nm)", fontsize=11)
    ax1.set_ylabel("Potencial / Energia (eV / V)", fontsize=11)
    ax1.set_title("Estrutura de Potencial Quântico em GAAFET 2 nm", fontsize=12)
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend()

    # 2. Velocidade Balística e Transmissão
    ax2.plot(x_eval * 1e9, vel_balistica * 1e-5, "darkgreen", lw=2.2, label=r"Velocidade Balística ($10^7\text{ cm/s}$)")
    ax2.set_xlabel("Posição no Canal $x$ (nm)", fontsize=11)
    ax2.set_ylabel(r"Velocidade ($10^7\text{ cm/s}$)", fontsize=11)
    ax2.set_title("Aceleração Quase-Balística de Portadores", fontsize=12)
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    caminho_grafico = "pinn_ito_gaafet_2nm_transporte.png"
    plt.savefig(caminho_grafico, dpi=300, bbox_inches="tight")
    print(f"\nGráfico científico salvo com sucesso em: {caminho_grafico}")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    modelo_treinado = treinar_pinn_ito_completa(epocas=300)
    avaliar_e_salvar_resultados(modelo_treinado)