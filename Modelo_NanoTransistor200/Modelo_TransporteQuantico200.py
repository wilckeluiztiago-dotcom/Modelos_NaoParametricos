"""
===============================================================================
Solução de EDEs de Transporte Quântico via Redes Neurais Físico-Informadas (PINNs)
em Nanotransistores 3D Gate-All-Around (GAAFET)
Autor: Luiz Tiago Wilcke
===============================================================================

Este módulo implementa:
  1. Acoplamento não-linear auto-consistente de Poisson 3D.
  2. Correção quântica por potencial de Bohm (Gradiente de Densidade).
  3. Equação estocástica de continuidade e transporte (Drift-Diffusion com flutuações).
  4. Rede Neural Físico-Informada (PINN) com projeção de Fourier espacial.
  5. Extração e análise Monte Carlo das curvas de transferência ID x VG.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# =============================================================================
# 1. CONSTANTES FÍSICAS E PARÂMETROS DO DISPOSITIVO (Sistema Internacional)
# =============================================================================
CARGA_ELEMENTAR = 1.60217663e-19           # Carga elementar q (C)
PERMISSIVIDADE_VACUO = 8.85418781e-12      # Permissividade do vácuo \varepsilon_0 (F/m)
PERMISSIVIDADE_SILICIO = 11.7 * PERMISSIVIDADE_VACUO  # Permissividade do Si (F/m)
PERMISSIVIDADE_OXIDO = 3.9 * PERMISSIVIDADE_VACUO     # Permissividade equivalente do dielétrico (F/m)
CONSTANTE_BOLTZMANN = 1.380649e-23         # Constante de Boltzmann k_B (J/K)
TEMPERATURA_OPERACAO = 300.0               # Temperatura de operação T (K)
TENSAO_TERMICA = (CONSTANTE_BOLTZMANN * TEMPERATURA_OPERACAO) / CARGA_ELEMENTAR  # V_T ~ 0.0259 V
CONSTANTE_PLANCK_REDUZIDA = 1.054571817e-34  # Constante de Dirac \hbar (J.s)
MASSA_EFETIVA_ELETRON = 0.26 * 9.10938356e-31  # Massa efetiva de condução m_n^* (kg)
MOBILIDADE_ELETRONS = 0.04                 # Mobilidade efetiva no canal \mu_n (m^2/V.s)
COEFICIENTE_DIFUSAO = MOBILIDADE_ELETRONS * TENSAO_TERMICA  # Relação de Einstein D_n (m^2/s)
CONCENTRACAO_INTRINSECA = 1.5e16           # Densidade intrínseca n_i (m^-3)
DOPAGEM_FONTE_DRENO = 1e26                 # Dopagem de contato n+ (m^-3)
DOPAGEM_CANAL = 1e21                      # Dopagem residual do canal n- (m^-3)

# Geometria do Nanotransistor GAAFET 3D
COMPRIMENTO_CANAL = 20e-9                  # Comprimento L (m)
LARGURA_NANOFIO = 5e-9                     # Largura W (m)
ALTURA_NANOFIO = 5e-9                      # Altura H (m)
ESPESSURA_OXIDO = 1.5e-9                   # Espessura t_ox (m)

DISPOSITIVO_COMPUTACIONAL = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. ARQUITETURA DA REDE NEURAL FÍSICO-INFORMADA (PINN)
# =============================================================================
class RedePINNTransporteQuanticoGAAFET(nn.Module):
    """
    Rede Neural Físico-Informada com Mapeamento de Fourier para resolução 
    de escalas microscópicas e fortes gradientes de potencial.
    Entrada: (x, y, z, V_porta, xi_estocastico)
    Saída: [Potencial Eletrostático \phi, Quase-Nível de Fermi \Phi_n]
    """
    def __init__(
        self, 
        dimensao_entrada: int = 5, 
        dimensao_saida: int = 2, 
        neuronios_ocultos: int = 128, 
        numero_camadas: int = 5
    ) -> None:
        super().__init__()
        
        # Matriz aleatória fixa para projeção de Fourier
        self.dimensao_fourier = 64
        self.matriz_projecao = nn.Parameter(
            torch.randn(dimensao_entrada, self.dimensao_fourier // 2) * 2.0, 
            requires_grad=False
        )
        
        camadas = [
            nn.Linear(self.dimensao_fourier, neuronios_ocultos),
            nn.SiLU()
        ]
        
        for _ in range(numero_camadas - 1):
            camadas.append(nn.Linear(neuronios_ocultos, neuronios_ocultos))
            camadas.append(nn.SiLU())
            
        self.bloco_oculto = nn.Sequential(*camadas)
        self.camada_saida = nn.Linear(neuronios_ocultos, dimensao_saida)

    def forward(self, entradas: torch.Tensor) -> torch.Tensor:
        projecao = 2.0 * np.pi * torch.matmul(entradas, self.matriz_projecao)
        caracteristicas_fourier = torch.cat([torch.sin(projecao), torch.cos(projecao)], dim=-1)
        ativacao_oculta = self.bloco_oculto(caracteristicas_fourier)
        saida_predicao = self.camada_saida(ativacao_oculta)
        return saida_predicao


# =============================================================================
# 3. MOTOR DIFERENCIAL DE RESÍDUOS EDP E TRANSPORTE QUÂNTICO
# =============================================================================
def calcular_residuos_edp_quantica(
    modelo: nn.Module, 
    coordenadas_dominio: torch.Tensor, 
    campo_permissividade: torch.Tensor, 
    campo_dopagem: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcula os resíduos exatos via diferenciação automática (Autograd):
      1. Equação Eletrostática de Poisson 3D não-linear.
      2. Equação de Transporte com Correção Quântica de Bohm (Density Gradient).
      3. Divergência de Corrente Estocástica estacionária (\nabla \cdot J = 0).
    """
    coordenadas_dominio.requires_grad_(True)
    predicoes = modelo(coordenadas_dominio)
    
    potencial_eletrostatico = predicoes[:, 0:1]
    quase_fermi = predicoes[:, 1:2]
    termo_estocastico_ler = coordenadas_dominio[:, 4:5]

    # Gradientes de primeira ordem para o potencial eletrostático
    gradiente_potencial = torch.autograd.grad(
        potencial_eletrostatico, 
        coordenadas_dominio, 
        grad_outputs=torch.ones_like(potencial_eletrostatico),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    dphi_dx = gradiente_potencial[:, 0:1]
    dphi_dy = gradiente_potencial[:, 1:2]
    dphi_dz = gradiente_potencial[:, 2:3]

    # Laplaciano 3D: \nabla^2 \phi
    d2phi_dx2 = torch.autograd.grad(dphi_dx, coordenadas_dominio, grad_outputs=torch.ones_like(dphi_dx), create_graph=True, retain_graph=True)[0][:, 0:1]
    d2phi_dy2 = torch.autograd.grad(dphi_dy, coordenadas_dominio, grad_outputs=torch.ones_like(dphi_dy), create_graph=True, retain_graph=True)[0][:, 1:2]
    d2phi_dz2 = torch.autograd.grad(dphi_dz, coordenadas_dominio, grad_outputs=torch.ones_like(dphi_dz), create_graph=True, retain_graph=True)[0][:, 2:3]
    laplaciano_potencial = d2phi_dx2 + d2phi_dy2 + d2phi_dz2

    # Concentração clássica e cálculo do Potencial Quântico de Bohm
    exponente_termo = torch.clamp((potencial_eletrostatico - quase_fermi) / TENSAO_TERMICA, min=-30.0, max=30.0)
    densidade_classica = CONCENTRACAO_INTRINSECA * torch.exp(exponente_termo)
    raiz_densidade = torch.sqrt(torch.clamp(densidade_classica, min=1e6))

    gradiente_raiz_densidade = torch.autograd.grad(
        raiz_densidade, 
        coordenadas_dominio, 
        grad_outputs=torch.ones_like(raiz_densidade),
        create_graph=True, 
        retain_graph=True
    )[0]

    d2_raiz_dx2 = torch.autograd.grad(gradiente_raiz_densidade[:, 0:1], coordenadas_dominio, grad_outputs=torch.ones_like(raiz_densidade), create_graph=True, retain_graph=True)[0][:, 0:1]
    d2_raiz_dy2 = torch.autograd.grad(gradiente_raiz_densidade[:, 1:2], coordenadas_dominio, grad_outputs=torch.ones_like(raiz_densidade), create_graph=True, retain_graph=True)[0][:, 1:2]
    d2_raiz_dz2 = torch.autograd.grad(gradiente_raiz_densidade[:, 2:3], coordenadas_dominio, grad_outputs=torch.ones_like(raiz_densidade), create_graph=True, retain_graph=True)[0][:, 2:3]
    
    laplaciano_raiz_densidade = d2_raiz_dx2 + d2_raiz_dy2 + d2_raiz_dz2
    
    potencial_bohm = (CONSTANTE_PLANCK_REDUZIDA**2 / (6.0 * MASSA_EFETIVA_ELETRON * CARGA_ELEMENTAR)) * (
        laplaciano_raiz_densidade / (raiz_densidade + 1e-12)
    )

    # Densidade de portadores quânticos corrigida
    densidade_quantica = CONCENTRACAO_INTRINSECA * torch.exp(
        torch.clamp((potencial_eletrostatico - quase_fermi + potencial_bohm) / TENSAO_TERMICA, min=-30.0, max=30.0)
    )

    # 1. Resíduo de Poisson: \nabla^2 \phi + q(N_D - n) / \varepsilon = 0
    residuo_poisson = laplaciano_potencial + (CARGA_ELEMENTAR * (campo_dopagem - densidade_quantica)) / campo_permissividade

    # 2. Resíduo de Continuidade e Transporte: \nabla \cdot J_n = 0
    potencial_propulsor = potencial_eletrostatico - quase_fermi + potencial_bohm
    gradiente_propulsor = torch.autograd.grad(
        potencial_propulsor, 
        coordenadas_dominio, 
        grad_outputs=torch.ones_like(potencial_propulsor),
        create_graph=True, 
        retain_graph=True
    )[0]

    densidade_corrente_x = CARGA_ELEMENTAR * MOBILIDADE_ELETRONS * densidade_quantica * gradiente_propulsor[:, 0:1] + 0.05 * termo_estocastico_ler * densidade_quantica
    densidade_corrente_y = CARGA_ELEMENTAR * MOBILIDADE_ELETRONS * densidade_quantica * gradiente_propulsor[:, 1:2]
    densidade_corrente_z = CARGA_ELEMENTAR * MOBILIDADE_ELETRONS * densidade_quantica * gradiente_propulsor[:, 2:3]

    dj_dx = torch.autograd.grad(densidade_corrente_x, coordenadas_dominio, grad_outputs=torch.ones_like(densidade_corrente_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    dj_dy = torch.autograd.grad(densidade_corrente_y, coordenadas_dominio, grad_outputs=torch.ones_like(densidade_corrente_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    dj_dz = torch.autograd.grad(densidade_corrente_z, coordenadas_dominio, grad_outputs=torch.ones_like(densidade_corrente_z), create_graph=True, retain_graph=True)[0][:, 2:3]

    residuo_continuidade = dj_dx + dj_dy + dj_dz

    return residuo_poisson, residuo_continuidade, densidade_corrente_z


# =============================================================================
# 4. AMOSTRADOR DE PONTOS DE COLOCAÇÃO E CONDIÇÕES DE CONTORNO
# =============================================================================
def gerar_amostras_treinamento_gaafet(
    numero_colocacao: int = 4096, 
    numero_fronteira: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Gera coordenadas espaciais 3D aleatórias no canal, nas interfaces
    e nos contatos de Fonte, Dreno e Porta envolvente (Gate-All-Around).
    """
    # Pontos de Colocação no Interior do Canal (0 <= x <= W, 0 <= y <= H, 0 <= z <= L)
    coord_x_interior = torch.rand(numero_colocacao, 1) * LARGURA_NANOFIO
    coord_y_interior = torch.rand(numero_colocacao, 1) * ALTURA_NANOFIO
    coord_z_interior = torch.rand(numero_colocacao, 1) * COMPRIMENTO_CANAL
    tensao_porta_interior = torch.rand(numero_colocacao, 1) * 1.0
    termo_estocastico_interior = torch.randn(numero_colocacao, 1)

    coordenadas_interior = torch.cat([
        coord_x_interior, coord_y_interior, coord_z_interior, 
        tensao_porta_interior, termo_estocastico_interior
    ], dim=1).to(DISPOSITIVO_COMPUTACIONAL)

    permissividade_interior = torch.full((numero_colocacao, 1), PERMISSIVIDADE_SILICIO).to(DISPOSITIVO_COMPUTACIONAL)
    
    # Perfil de dopagem abrupto n+/intrínseco/n+
    perfil_dopagem = torch.where(
        (coord_z_interior < 4e-9) | (coord_z_interior > 16e-9),
        torch.full_like(coord_z_interior, DOPAGEM_FONTE_DRENO),
        torch.full_like(coord_z_interior, DOPAGEM_CANAL)
    ).to(DISPOSITIVO_COMPUTACIONAL)

    # Condição de Contorno: Fonte (z = 0, \phi = 0, \Phi_n = 0)
    contorno_fonte = torch.cat([
        torch.rand(numero_fronteira, 1) * LARGURA_NANOFIO,
        torch.rand(numero_fronteira, 1) * ALTURA_NANOFIO,
        torch.zeros(numero_fronteira, 1),
        torch.rand(numero_fronteira, 1) * 1.0,
        torch.randn(numero_fronteira, 1)
    ], dim=1).to(DISPOSITIVO_COMPUTACIONAL)

    # Condição de Contorno: Dreno (z = L, \phi = V_D, \Phi_n = V_D)
    contorno_dreno = torch.cat([
        torch.rand(numero_fronteira, 1) * LARGURA_NANOFIO,
        torch.rand(numero_fronteira, 1) * ALTURA_NANOFIO,
        torch.full((numero_fronteira, 1), COMPRIMENTO_CANAL),
        torch.rand(numero_fronteira, 1) * 1.0,
        torch.randn(numero_fronteira, 1)
    ], dim=1).to(DISPOSITIVO_COMPUTACIONAL)

    # Condição de Contorno: Porta Envolvente (GAA - Superfícies Laterais x = 0 ou W, y = 0 ou H)
    coord_x_porta = torch.randint(0, 2, (numero_fronteira, 1)).float() * LARGURA_NANOFIO
    coord_y_porta = torch.rand(numero_fronteira, 1) * ALTURA_NANOFIO
    coord_z_porta = torch.rand(numero_fronteira, 1) * COMPRIMENTO_CANAL
    tensao_porta_g = torch.rand(numero_fronteira, 1) * 1.0
    
    contorno_porta = torch.cat([
        coord_x_porta, coord_y_porta, coord_z_porta, 
        tensao_porta_g, torch.randn(numero_fronteira, 1)
    ], dim=1).to(DISPOSITIVO_COMPUTACIONAL)

    return {
        "interior": coordenadas_interior,
        "permissividade": permissividade_interior,
        "dopagem": perfil_dopagem,
        "fonte": contorno_fonte,
        "dreno": contorno_dreno,
        "porta": contorno_porta
    }


# =============================================================================
# 5. MOTOR DE TREINAMENTO DA PINN
# =============================================================================
def treinar_pinn_nanotransistor(epocas: int = 600) -> nn.Module:
    """Treina a rede informada pela física via otimizador AdamW e cosseno adaptativo."""
    modelo_pinn = RedePINNTransporteQuanticoGAAFET().to(DISPOSITIVO_COMPUTACIONAL)
    otimizador = torch.optim.AdamW(modelo_pinn.parameters(), lr=1e-3, weight_decay=1e-5)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas)
    
    print("=" * 80)
    print("INICIALIZANDO TREINAMENTO DA PINN DE TRANSPORTE QUÂNTICO (GAAFET 3D)")
    print(f"Autor: Luiz Tiago Wilcke | Dispositivo: {DISPOSITIVO_COMPUTACIONAL}")
    print("=" * 80)

    for epoca in range(1, epocas + 1):
        otimizador.zero_grad()
        amostras = gerar_amostras_treinamento_gaafet()
        
        # 1. Resíduos Diferenciais no Domínio
        res_poisson, res_continuidade, _ = calcular_residuos_edp_quantica(
            modelo_pinn, amostras["interior"], amostras["permissividade"], amostras["dopagem"]
        )
        
        perda_poisson = torch.mean(res_poisson**2) * 1e-28
        perda_continuidade = torch.mean(res_continuidade**2) * 1e-12

        # 2. Resíduos nas Condições de Contorno
        saida_fonte = modelo_pinn(amostras["fonte"])
        perda_fonte = torch.mean((saida_fonte[:, 0:1] - 0.0)**2) + torch.mean((saida_fonte[:, 1:2] - 0.0)**2)

        saida_dreno = modelo_pinn(amostras["dreno"])
        tensao_dreno_nominal = 0.7
        perda_dreno = torch.mean((saida_dreno[:, 0:1] - tensao_dreno_nominal)**2) + torch.mean((saida_dreno[:, 1:2] - tensao_dreno_nominal)**2)

        saida_porta = modelo_pinn(amostras["porta"])
        tensao_porta_alvo = amostras["porta"][:, 3:4]
        perda_porta = torch.mean((saida_porta[:, 0:1] - tensao_porta_alvo)**2)

        perda_total = perda_poisson + perda_continuidade + 10.0 * (perda_fonte + perda_dreno + perda_porta)
        
        perda_total.backward()
        otimizador.step()
        agendador.step()

        if epoca % 100 == 0 or epoca == 1:
            print(
                f"Época {epoca:4d}/{epocas} | "
                f"Perda Total: {perda_total.item():.6e} | "
                f"Poisson: {perda_poisson.item():.4e} | "
                f"Continuidade: {perda_continuidade.item():.4e}"
            )

    print("=" * 80)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO.")
    print("=" * 80)
    return modelo_pinn


# =============================================================================
# 6. EXTRAÇÃO ELETROSTÁTICA E MONTE CARLO DAS CURVAS ID x VG
# =============================================================================
def extrair_curvas_caracteristicas_id_vg(
    modelo_pinn: nn.Module, 
    pontos_tensao_porta: int = 25, 
    amostras_monte_carlo: int = 30
) -> None:
    """
    Integra a densidade de corrente de dreno em 2D na seção transversal terminal,
    gerando curvas de transferência com bandas de incerteza estocástica (+- 2 sigma).
    """
    modelo_pinn.eval()
    valores_tensao_porta = np.linspace(0.0, 1.0, pontos_tensao_porta)
    matriz_corrente_mc = np.zeros((amostras_monte_carlo, pontos_tensao_porta))
    
    # Grade de integração na seção transversal do dreno (z = COMPRIMENTO_CANAL)
    resolucao_grade = 16
    malha_x = torch.linspace(0, LARGURA_NANOFIO, resolucao_grade)
    malha_y = torch.linspace(0, ALTURA_NANOFIO, resolucao_grade)
    grade_x, grade_y = torch.meshgrid(malha_x, malha_y, indexing="ij")
    
    x_vetor = grade_x.reshape(-1, 1)
    y_vetor = grade_y.reshape(-1, 1)
    z_vetor = torch.full_like(x_vetor, COMPRIMENTO_CANAL)
    elemento_area = (LARGURA_NANOFIO / resolucao_grade) * (ALTURA_NANOFIO / resolucao_grade)

    for idx_mc in range(amostras_monte_carlo):
        realizacao_estocastica = np.random.normal(0.0, 1.0)
        for idx_vg, valor_vg in enumerate(valores_tensao_porta):
            pontos_avaliacao = torch.cat([
                x_vetor, y_vetor, z_vetor,
                torch.full_like(x_vetor, valor_vg),
                torch.full_like(x_vetor, realizacao_estocastica)
            ], dim=1).to(DISPOSITIVO_COMPUTACIONAL)
            
            permissividade_aval = torch.full((pontos_avaliacao.shape[0], 1), PERMISSIVIDADE_SILICIO).to(DISPOSITIVO_COMPUTACIONAL)
            dopagem_aval = torch.full((pontos_avaliacao.shape[0], 1), DOPAGEM_FONTE_DRENO).to(DISPOSITIVO_COMPUTACIONAL)
            
            _, _, densidade_corrente_z = calcular_residuos_edp_quantica(
                modelo_pinn, pontos_avaliacao, permissividade_aval, dopagem_aval
            )
            
            # Integral numérica: I_D = \iint J_z dx dy
            corrente_total = torch.sum(torch.abs(densidade_corrente_z)).item() * elemento_area
            matriz_corrente_mc[idx_mc, idx_vg] = max(corrente_total, 1e-15)

    corrente_media_dreno = np.mean(matriz_corrente_mc, axis=0)
    desvio_padrao_corrente = np.std(matriz_corrente_mc, axis=0)

    # Visualização Científica dos Resultados
    fig, (eixo_log, eixo_linear) = plt.subplots(1, 2, figsize=(13, 5))

    # Curva em Escala Logarítmica (Avaliação de Subthreshold Swing e Corrente OFF)
    eixo_log.plot(valores_tensao_porta, corrente_media_dreno, color="royalblue", lw=2.2, label=r"PINN Média $\mathbb{E}[I_D]$")
    eixo_log.fill_between(
        valores_tensao_porta, 
        np.maximum(corrente_media_dreno - 2 * desvio_padrao_corrente, 1e-15), 
        corrente_media_dreno + 2 * desvio_padrao_corrente, 
        color="royalblue", alpha=0.25, label=r"Dispersão Quântica ($\pm 2\sigma$)"
    )
    eixo_log.set_yscale("log")
    eixo_log.set_xlabel("Tensão de Porta $V_G$ (V)", fontsize=11)
    eixo_log.set_ylabel("Corrente de Dreno $I_D$ (A)", fontsize=11)
    eixo_log.set_title("Curva de Transferência $I_D \\times V_G$ (Escala Log)", fontsize=12)
    eixo_log.grid(True, which="both", ls="--", alpha=0.5)
    eixo_log.legend()

    # Curva em Escala Linear (Avaliação de Corrente ON e Condução)
    eixo_linear.plot(valores_tensao_porta, corrente_media_dreno * 1e6, color="crimson", lw=2.2, label=r"PINN Média $\mathbb{E}[I_D]$")
    eixo_linear.fill_between(
        valores_tensao_porta, 
        (corrente_media_dreno - 2 * desvio_padrao_corrente) * 1e6, 
        (corrente_media_dreno + 2 * desvio_padrao_corrente) * 1e6, 
        color="crimson", alpha=0.25, label=r"Dispersão Quântica ($\pm 2\sigma$)"
    )
    eixo_linear.set_xlabel("Tensão de Porta $V_G$ (V)", fontsize=11)
    eixo_linear.set_ylabel("Corrente de Dreno $I_D$ ($\mu$A)", fontsize=11)
    eixo_linear.set_title("Curva de Transferência $I_D \\times V_G$ (Escala Linear)", fontsize=12)
    eixo_linear.grid(True, ls="--", alpha=0.5)
    eixo_linear.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    modelo_treinado = treinar_pinn_nanotransistor(epocas=600)
    extrair_curvas_caracteristicas_id_vg(modelo_treinado, pontos_tensao_porta=25, amostras_monte_carlo=30)