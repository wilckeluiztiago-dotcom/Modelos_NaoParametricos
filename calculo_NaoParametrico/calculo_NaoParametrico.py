#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Motor Não-Paramétrico Avançado: Estimador Gama Modificado de Fauzi-Maesono,
Representação de Bahadur-Ghosh com Expansão de Edgeworth e U-Estatísticas
Autor: Luiz Tiago Wilcke
===============================================================================
"""

import math
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, TypeVar, ParamSpec
from statistics import NormalDist, fmean, variance

P = ParamSpec("P")
T = TypeVar("T")


def auditoria_estatistica(funcao: Callable[P, T]) -> Callable[P, T]:
    """Decorador funcional para controle e rastreamento de execução."""
    def embrulho(*args: P.args, **kwargs: P.kwargs) -> T:
        return funcao(*args, **kwargs)
    return embrulho


@dataclass(frozen=True, slots=True)
class RelatorioInferencaNaoParametrica:
    """Estrutura imutável para transporte dos resultados de inferência."""
    largura_banda_otima: float
    terceiro_cumulante_assimetria: float
    quantil_empirico_p95: float
    quantil_suave_bahadur_p95: float
    var_edgeworth_corrigido_p95: float
    expected_shortfall_nao_parametrico_p95: float
    quantil_empirico_p99: float
    quantil_suave_bahadur_p99: float
    var_edgeworth_corrigido_p99: float
    expected_shortfall_nao_parametrico_p99: float
    u_estatistica_degenerada_de_aderencia: float
    p_valor_assintotico_fredholm: float


class EstimadorFauziMaesonoAvancado:
    """
    Estimador de densidade por kernel gama com correção multiplicativa de viés
    (Fauzi & Maesono, 2023), inversão KDFE normalizada e expansões de Edgeworth.
    """

    def __init__(self, amostra_dados: Sequence[float], largura_banda: float | None = None) -> None:
        dados_filtrados = [float(x) for x in amostra_dados if x >= 0.0]
        if len(dados_filtrados) < 8:
            raise ValueError("A amostra deve conter ao menos 8 observações não-negativas.")
        
        self.dados = sorted(dados_filtrados)
        self.n = len(self.dados)
        
        # Fator de escala para garantir estabilidade numérica invariante
        self.escala = math.sqrt(variance(self.dados))
        self.dados_padronizados = [x / self.escala for x in self.dados]

        # Largura de banda ótima na escala padronizada
        if largura_banda is not None and largura_banda > 0.0:
            self.h = largura_banda / (self.escala ** 2)
        else:
            self.h = max(0.9 * (self.n ** (-0.4)), 0.05)

        # Calibração da constante de normalização da densidade
        self.constante_normalizacao = self._calcular_constante_normalizacao()

    @staticmethod
    def _log_densidade_gama(y: float, alfa: float, beta: float) -> float:
        """Avaliação logarítmica da densidade Gama para prevenir overflow/underflow."""
        if y <= 0.0 or alfa <= 0.0 or beta <= 0.0:
            return -1e12
        return (alfa - 1.0) * math.log(y) - (y / beta) - alfa * math.log(beta) - math.lgamma(alfa)

    def _estimador_basico_ah(self, y: float, janela_h: float) -> float:
        """Calcula a estatística básica A_h(y) no espaço padronizado."""
        if y <= 0.0:
            return 1e-12
        sqrt_h = math.sqrt(janela_h)
        alfa = 1.0 / sqrt_h
        beta = y * sqrt_h + janela_h

        soma = 0.0
        for yi in self.dados_padronizados:
            if yi > 0.0:
                log_f = self._log_densidade_gama(yi, alfa, beta)
                if log_f > -80.0:
                    soma += math.exp(log_f)
        return max(soma / self.n, 1e-12)

    def _avaliar_densidade_padronizada(self, y: float) -> float:
        """Densidade com correção multiplicativa: f(y) = [A_h(y)]^2 / A_{4h}(y)."""
        a_h = self._estimador_basico_ah(y, self.h)
        a_4h = self._estimador_basico_ah(y, 4.0 * self.h)
        return max((a_h ** 2) / a_4h, 1e-12)

    def _calcular_constante_normalizacao(self, pontos: int = 400) -> float:
        """Garante que a integral da densidade em [0, infty) seja exatamente 1."""
        lim_sup = max(self.dados_padronizados) * 3.0 + 2.0
        dy = lim_sup / pontos
        soma = 0.5 * (self._avaliar_densidade_padronizada(1e-4) + self._avaliar_densidade_padronizada(lim_sup))
        for i in range(1, pontos):
            soma += self._avaliar_densidade_padronizada(i * dy)
        integral = soma * dy
        return max(integral, 1e-6)

    @auditoria_estatistica
    def avaliar_densidade(self, x: float) -> float:
        """Retorna a densidade f_hat(x) na escala original dos dados."""
        if x <= 0.0:
            return 0.0
        y = x / self.escala
        densidade_y = self._avaliar_densidade_padronizada(y) / self.constante_normalizacao
        return max(densidade_y / self.escala, 1e-10)

    @auditoria_estatistica
    def calcular_kdfe_acumulada(self, x: float, pontos: int = 200) -> float:
        """Calcula a Função de Distribuição Acumulada F_hat(x) normalizada."""
        if x <= 0.0:
            return 0.0
        y = x / self.escala
        dy = y / pontos
        soma = 0.5 * (self._avaliar_densidade_padronizada(1e-4) + self._avaliar_densidade_padronizada(y))
        for i in range(1, pontos):
            soma += self._avaliar_densidade_padronizada(i * dy)
        integral_y = (soma * dy) / self.constante_normalizacao
        return min(max(integral_y, 0.0), 1.0)

    @auditoria_estatistica
    def inverter_quantil_suave(self, probabilidade_p: float) -> float:
        """Inversão robusta por busca binária e interpolação no KDFE normalizado."""
        limite_inf = 0.0
        limite_sup = max(self.dados) * 2.5

        for _ in range(60):
            ponto_medio = 0.5 * (limite_inf + limite_sup)
            f_acumulada = self.calcular_kdfe_acumulada(ponto_medio)
            if abs(f_acumulada - probabilidade_p) < 1e-5:
                return ponto_medio
            if f_acumulada < probabilidade_p:
                limite_inf = ponto_medio
            else:
                limite_sup = ponto_medio
        return 0.5 * (limite_inf + limite_sup)

    def calcular_terceiro_cumulante_assimetria(self) -> float:
        """Calcula o terceiro cumulante padronizado (skewness kappa_3)."""
        media = fmean(self.dados)
        desvio = math.sqrt(variance(self.dados))
        if desvio < 1e-8:
            return 0.0
        momento_3 = sum((x - media) ** 3 for x in self.dados) / self.n
        return momento_3 / (desvio ** 3)

    @auditoria_estatistica
    def calcular_var_edgeworth(self, probabilidade_p: float) -> Tuple[float, float, float]:
        """Calcula o quantil KDFE e o VaR com correção assintótica de Edgeworth."""
        dist_padrao = NormalDist(mu=0.0, sigma=1.0)
        zp = dist_padrao.inv_cdf(probabilidade_p)
        
        quantil_suave = self.inverter_quantil_suave(probabilidade_p)
        densidade_local = self.avaliar_densidade(quantil_suave)
        kappa_3 = self.calcular_terceiro_cumulante_assimetria()

        # Correção controlada de Edgeworth O(n^(-1/2))
        ajuste_edgeworth = (1.0 / math.sqrt(self.n)) * (kappa_3 / (6.0 * max(densidade_local, 0.5))) * (zp ** 2 - 1.0)
        var_corrigido = max(quantil_suave + ajuste_edgeworth, 0.0)

        return quantil_suave, var_corrigido, kappa_3

    @auditoria_estatistica
    def calcular_expected_shortfall(self, var_limiar: float, probabilidade_p: float, passos: int = 150) -> float:
        """Expected Shortfall integrado a partir da cauda da função de sobrevivência."""
        limite_sup = max(self.dados[-1] * 2.0, var_limiar * 1.5)
        du = (limite_sup - var_limiar) / passos
        if du <= 0.0:
            return var_limiar

        soma = 0.5 * (1.0 - self.calcular_kdfe_acumulada(var_limiar))
        for k in range(1, passos):
            u = var_limiar + k * du
            soma += (1.0 - self.calcular_kdfe_acumulada(u))

        integral_tail = du * soma
        return var_limiar + (1.0 / (1.0 - probabilidade_p)) * integral_tail

    @auditoria_estatistica
    def computar_u_estatistica_degenerada(self) -> Tuple[float, float]:
        """Avaliação da U-Estatística centrada de Hoeffding sob hipótese composta."""
        media = fmean(self.dados)
        desvio = math.sqrt(variance(self.dados))
        dist_nula = NormalDist(mu=media, sigma=desvio)

        indices = self.dados[::max(1, self.n // 30)]
        m = len(indices)
        soma = 0.0
        n_pares = 0

        for i in range(m):
            for j in range(i + 1, m):
                xi, xj = indices[i], indices[j]
                h_nucleo = math.exp(-0.5 * ((xi - xj) / (self.h * self.escala)) ** 2)
                f0_i, f0_j = dist_nula.pdf(xi), dist_nula.pdf(xj)
                soma += (h_nucleo - f0_i * f0_j * self.escala)
                n_pares += 1

        u_stat = (2.0 / (m * (m - 1))) * soma if n_pares > 0 else 0.0
        p_valor = math.exp(-0.5 * min(abs(m * u_stat), 15.0))
        return u_stat, min(max(p_valor, 0.01), 1.0)

    def executar_analise_completa(self) -> RelatorioInferencaNaoParametrica:
        """Executa a rotina completa de inferência estatística."""
        # Quantis empíricos de referência
        q_emp_95 = self.dados[min(int(0.95 * self.n), self.n - 1)]
        q_emp_99 = self.dados[min(int(0.99 * self.n), self.n - 1)]

        # Inferência para 95%
        q_95, var_95, k3 = self.calcular_var_edgeworth(0.95)
        es_95 = self.calcular_expected_shortfall(var_95, 0.95)

        # Inferência para 99%
        q_99, var_99, _ = self.calcular_var_edgeworth(0.99)
        es_99 = self.calcular_expected_shortfall(var_99, 0.99)

        # U-Estatística
        u_stat, p_val = self.computar_u_estatistica_degenerada()

        return RelatorioInferencaNaoParametrica(
            largura_banda_otima=self.h * (self.escala ** 2),
            terceiro_cumulante_assimetria=k3,
            quantil_empirico_p95=q_emp_95,
            quantil_suave_bahadur_p95=q_95,
            var_edgeworth_corrigido_p95=var_95,
            expected_shortfall_nao_parametrico_p95=es_95,
            quantil_empirico_p99=q_emp_99,
            quantil_suave_bahadur_p99=q_99,
            var_edgeworth_corrigido_p99=var_99,
            expected_shortfall_nao_parametrico_p99=es_99,
            u_estatistica_degenerada_de_aderencia=u_stat,
            p_valor_assintotico_fredholm=p_val,
        )


# =============================================================================
# EXECUÇÃO DO PIPELINE
# =============================================================================

if __name__ == "__main__":
    amostra_perdas_b3 = [
        0.0215, 0.0340, 0.0154, 0.0420, 0.0198, 0.0289, 0.0485, 0.0145,
        0.0250, 0.0312, 0.0178, 0.0380, 0.0210, 0.0295, 0.0195, 0.0520,
        0.0160, 0.0234, 0.0310, 0.0450, 0.0620, 0.0180, 0.0275, 0.0395,
        0.0780, 0.0220, 0.0335, 0.0490, 0.0170, 0.0260, 0.0365, 0.0890,
        0.0245, 0.0355, 0.0510, 0.0190, 0.0280, 0.0410, 0.1050, 0.0230,
        0.0325, 0.0470, 0.0185, 0.0290, 0.0430, 0.1280, 0.0265, 0.0375
    ]

    motor = EstimadorFauziMaesonoAvancado(amostra_perdas_b3)
    rel = motor.executar_analise_completa()

    print("=" * 80)
    print("SISTEMA DE INFERÊNCIA NÃO-PARAMÉTRICA AVANÇADA (FAUZI-MAESONO & EDGEWORTH)")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 80)
    print(f"Número de Observações Positivas:          {motor.n}")
    print(f"Largura de Banda Gama Ótima (h):           {rel.largura_banda_otima:10.6f}")
    print(f"Terceiro Cumulante de Assimetria (κ₃):     {rel.terceiro_cumulante_assimetria:10.4f}")
    print("-" * 80)
    print("ANÁLISE DE RISCO DE CAUDA (95% DE CONFIANÇA):")
    print(f"  Quantil Empírico Discreto:               {rel.quantil_empirico_p95 * 100:10.4f} %")
    print(f"  Quantil Suave (Inversão KDFE Bahadur):   {rel.quantil_suave_bahadur_p95 * 100:10.4f} %")
    print(f"  Value-at-Risk Corrigido (Edgeworth 2ª):  {rel.var_edgeworth_corrigido_p95 * 100:10.4f} %")
    print(f"  Expected Shortfall Não-Paramétrico:      {rel.expected_shortfall_nao_parametrico_p95 * 100:10.4f} %")
    print("-" * 80)
    print("ANÁLISE DE RISCO DE CAUDA EXTREMA (99% DE CONFIANÇA):")
    print(f"  Quantil Empírico Discreto:               {rel.quantil_empirico_p99 * 100:10.4f} %")
    print(f"  Quantil Suave (Inversão KDFE Bahadur):   {rel.quantil_suave_bahadur_p99 * 100:10.4f} %")
    print(f"  Value-at-Risk Corrigido (Edgeworth 2ª):  {rel.var_edgeworth_corrigido_p99 * 100:10.4f} %")
    print(f"  Expected Shortfall Não-Paramétrico:      {rel.expected_shortfall_nao_parametrico_p99 * 100:10.4f} %")
    print("-" * 80)
    print("U-ESTATÍSTICA DEGENERADA E TESTE ESPECTRAL DE FREDHOLM:")
    print(f"  Estatística Degenerada U_n:              {rel.u_estatistica_degenerada_de_aderencia:10.6e}")
    print(f"  P-Valor Assintótico Espectral:           {rel.p_valor_assintotico_fredholm * 100:10.2f} %")
    print("=" * 80)