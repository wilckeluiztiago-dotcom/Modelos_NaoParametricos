"""
PINN-Ito-QuantumTransport
Estimação de Deriva Balística Quântica via Redes Neurais Físico-Informadas
e Cálculo de Itô em Nanotransistores GAAFET 3D Sub-3 nm

Autor: Luiz Tiago Wilcke (implementação fiel ao readme do repositório)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# ============================================================
# 1. Constantes físicas e parâmetros do dispositivo
# ============================================================
hbar = 1.0545718e-34          # J·s
m0 = 9.10938e-31              # kg
m_eff = 0.26 * m0             # massa efetiva Si
q = 1.60217662e-19            # C
kB = 1.380649e-23             # J/K
T = 300.0                     # K

# Dimensões do GAAFET (nm → m)
L = 12e-9                     # comprimento do canal
W = 3e-9
H = 3e-9
V_D = 0.65                    # V
V_G = 0.70                    # V
V_S = 0.0

# Energia da primeira sub-banda (eV)
E11 = (hbar**2 * np.pi**2 / (2 * m_eff) * (1/W**2 + 1/H**2)) / q
print(f"Energia da sub-banda E11 = {E11:.4f} eV")

# ============================================================
# 2. Rede Neural PINN
# ============================================================
class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 2]):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-2):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f"act_{i}", nn.Tanh())
        self.net.add_module("output", nn.Linear(layers[-2], layers[-1]))

    def forward(self, x, t):
        # x normalizado [0,1], t normalizado [0,1]
        inp = torch.cat([x, t], dim=1)
        out = self.net(inp)
        mu = out[:, 0:1]          # quase-nível de Fermi
        b  = out[:, 1:2]          # função de deriva
        return mu, b

# ============================================================
# 3. Funções de perda
# ============================================================
def schrodinger_residual(model, x, t, Vgate):
    """Resíduo da equação de Schrödinger unidimensional efetiva"""
    x.requires_grad_(True)
    t.requires_grad_(True)
    mu, _ = model(x, t)

    # derivadas
    dmu_dt = grad(mu, t, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
    dmu_dx = grad(mu, x, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
    d2mu_dx2 = grad(dmu_dx, x, grad_outputs=torch.ones_like(dmu_dx), create_graph=True)[0]

    # potencial efetivo (aproximação)
    Veff = q * Vgate + E11 - mu          # em eV-ish (unidades normalizadas)

    # residual (unidades normalizadas para estabilidade numérica)
    residual = dmu_dt - (-0.5 * d2mu_dx2 + Veff * mu)
    return residual

def ito_contrast(model, x, t, N_paths=32):
    """Funcional de contraste empírico de Itô (simplificado)"""
    mu, b = model(x, t)
    # Aproximação do contraste de mínimos quadrados
    # γ_N(b) ≈ (1/NT) ∫ [b² ds - 2 b dμ]
    contrast = torch.mean(b**2) - 2.0 * torch.mean(b * mu)
    return contrast

def boundary_loss(model):
    """Condições de contorno na Fonte e Dreno"""
    # Fonte x=0
    x_s = torch.zeros(50, 1)
    t_s = torch.rand(50, 1)
    mu_s, _ = model(x_s, t_s)
    loss_s = torch.mean((mu_s - V_S)**2)

    # Dreno x=1 (normalizado)
    x_d = torch.ones(50, 1)
    t_d = torch.rand(50, 1)
    mu_d, _ = model(x_d, t_d)
    loss_d = torch.mean((mu_d - V_D)**2)

    return loss_s + loss_d

# ============================================================
# 4. Treinamento
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# pesos da perda multiobjetivo
w_quant = 1.0
w_ito   = 0.1
w_cont  = 10.0

n_epochs = 3000
print("\nIniciando treinamento da PINN-Itô...")

for epoch in range(n_epochs):
    # pontos de colocação
    x = torch.rand(256, 1, device=device)
    t = torch.rand(256, 1, device=device)

    # potencial de gate (simples linear + barreira)
    Vgate = V_G * (1.0 - 0.3 * torch.sin(np.pi * x))   # forma suave

    # perdas
    res = schrodinger_residual(model, x, t, Vgate)
    L_sch = torch.mean(res**2)
    L_ito = ito_contrast(model, x, t)
    L_bc  = boundary_loss(model)

    loss = w_quant * L_sch + w_ito * L_ito + w_cont * L_bc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:4d} | Loss total: {loss.item():.4e} | "
              f"Schr: {L_sch.item():.3e} | Itô: {L_ito.item():.3e} | BC: {L_bc.item():.3e}")

print("Treinamento concluído.\n")

# ============================================================
# 5. Extração de métricas balísticas
# ============================================================
def extract_metrics(model, n_points=11):
    x_nm = np.linspace(0, 12, n_points)          # nm
    x_norm = torch.tensor(x_nm / 12.0, dtype=torch.float32).view(-1, 1)
    t_fixed = torch.ones_like(x_norm) * 0.5      # instante intermediário

    with torch.no_grad():
        mu, b = model(x_norm, t_fixed)
        mu = mu.numpy().flatten()
        b  = b.numpy().flatten()

    # Velocidade balística (cm/s)
    # v = sqrt(2 q μ / m_eff)
    v_ball = np.sqrt(2 * q * np.maximum(mu, 0) / m_eff) * 100   # m/s → cm/s
    v_ball_1e7 = v_ball / 1e7

    # Transmissão de Landauer-Büttiker (WKB aproximado)
    T_landauer = np.zeros_like(mu)
    for i, xx in enumerate(x_nm):
        # barreira efetiva
        barrier = np.maximum(E11 + V_G - mu[i], 0)
        kappa = np.sqrt(2 * m_eff * barrier * q) / hbar
        # integral simplificada
        integral = kappa * (L * (1 - xx/12))   # resto do canal
        T_landauer[i] = np.exp(-2 * integral)
        T_landauer[i] = np.clip(T_landauer[i], 0, 1)

    return x_nm, mu, v_ball_1e7, T_landauer

x_nm, mu, v_ball, T = extract_metrics(model)

# ============================================================
# 6. Tabela de resultados
# ============================================================
print("=" * 70)
print(f"{'Posição x (nm)':>14} | {'μ(x) (V)':>10} | {'Vel. Balística (10^7 cm/s)':>26} | {'T Landauer':>12}")
print("-" * 70)
for i in range(len(x_nm)):
    print(f"{x_nm[i]:14.2f} | {mu[i]:10.4f} | {v_ball[i]:26.4f} | {T[i]:12.4f}")
print("=" * 70)

# ============================================================
# 7. Gráficos
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].plot(x_nm, mu, 'b-o', linewidth=2)
axes[0].set_xlabel("Posição x (nm)")
axes[0].set_ylabel("Quase-Fermi μ(x) (V)")
axes[0].set_title("Quase-nível de Fermi")
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_nm, v_ball, 'r-s', linewidth=2)
axes[1].set_xlabel("Posição x (nm)")
axes[1].set_ylabel("Velocidade Balística (10⁷ cm/s)")
axes[1].set_title("Velocidade de Injeção Balística")
axes[1].grid(True, alpha=0.3)

axes[2].plot(x_nm, T, 'g-^', linewidth=2)
axes[2].set_xlabel("Posição x (nm)")
axes[2].set_ylabel("Transmissão Landauer T")
axes[2].set_title("Coeficiente de Transmissão")
axes[2].set_ylim(0, 1.05)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultados_pinn_ito_gaafet.png", dpi=150)
plt.show()

print("\nGráfico salvo em: resultados_pinn_ito_gaafet.png")
print("Simulação finalizada com sucesso.")
