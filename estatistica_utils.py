"""
Funções auxiliares para o Trabalho de Estatística Aplicada:
Amostragem, Intervalo de Confiança e Teste de Hipóteses (Normal Z e t-Student).
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from IPython.display import display, Markdown

# Padrão visual limpo para os gráficos da apresentação
plt.rcParams["figure.figsize"] = (10, 4.8)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 11
pd.set_option("display.precision", 4)


# ==============================================================================
# UTILITÁRIOS DE FORMATAÇÃO E FLUXOGRAMA (Z vs. t-STUDENT)
# ==============================================================================
def fmt_tex(valor: float, casas: int = 4) -> str:
    """Formata número decimal com vírgula sem espaçamento extra no LaTeX (ex.: 0{,}5374)."""
    partes = f"{valor:.{casas}f}".split(".")
    return f"{partes[0]}{{,}}{partes[1]}" if len(partes) > 1 else partes[0]


def fmt_br(valor: float, casas: int = 4) -> str:
    """Formata número decimal no padrão brasileiro para textos e gráficos."""
    return f"{valor:.{casas}f}".replace(".", ",")


def escolher_modelo(n: int, sigma_conhecido: bool = False) -> dict:
    """
    Fluxograma oficial da disciplina:
    1. σ(x) conhecido -> Normal (Z)
    2. σ(x) desconhecido e n > 30 -> aproxima S(x) -> σ(x) e usa Normal (Z)
    3. σ(x) desconhecido e n <= 30 -> t-Student (t) com S(x) (Bessel) e gl = n - 1
    """
    if sigma_conhecido:
        return {
            "dist": "Z",
            "gl": None,
            "regra": "σ(x) populacional conhecido → **Distribuição Normal (Z)**.",
        }
    if n > 30:
        return {
            "dist": "Z",
            "gl": None,
            "regra": f"σ(x) desconhecido e $n = {n} > 30$ → aproxima $S(x) \\to \\sigma(x)$ e usa **Distribuição Normal ($Z$)**.",
        }
    gl = n - 1
    return {
        "dist": "t",
        "gl": gl,
        "regra": f"σ(x) desconhecido e $n = {n} \\le 30$ → **Distribuição $t\\text{{-Student}}$ ($t$)** com $S(x)$ (Bessel) e $gl = n - 1 = {gl}$.",
    }


# ==============================================================================
# PARTE 1: RAIO-X, AMOSTRAGEM, INTERVALO DE CONFIANÇA E TAMANHO DA AMOSTRA
# ==============================================================================
def carregar_dados(caminho: str = "data/vgsales.csv") -> pd.DataFrame:
    """Carrega o dataset de vendas de jogos."""
    return pd.read_csv(caminho)


def exibir_raio_x(df: pd.DataFrame, coluna: str = "Global_Sales") -> tuple[float, float]:
    """Exibe parâmetros populacionais (ddof=0) e o Top 5 jogos (outliers)."""
    serie = df[coluna].dropna()
    n_pop = len(serie)
    mu_pop = float(serie.mean())
    var_pop = float(serie.var(ddof=0))
    sigma_pop = float(serie.std(ddof=0))

    resumo = pd.DataFrame([{
        "População (N)": n_pop,
        "Média Pop. μ (mi)": mu_pop,
        "Variância Pop. σ²(x) (ddof=0)": var_pop,
        "Desvio Pop. σ(x) (ddof=0)": sigma_pop,
        "Mediana (mi)": float(serie.median()),
        "Máximo (mi)": float(serie.max()),
    }])

    display(Markdown("### 📌 Parâmetros Populacionais (`Global_Sales` em milhões de cópias)"))
    display(resumo.set_index("População (N)").round(4))

    display(Markdown("### 🏆 Top 5 Jogos Mais Vendidos (Outliers extremos da cauda direita)"))
    cols_top = ["Rank", "Name", "Platform", "Year", "Genre", "Publisher", "Global_Sales"]
    top5 = df[cols_top].head(5).copy()
    top5["Year"] = top5["Year"].astype(int)
    display(top5.set_index("Rank"))

    return mu_pop, sigma_pop


def amostrar_estratificada_genero(df: pd.DataFrame, n: int = 60, semente: int = 42) -> pd.DataFrame:
    """Extrai Amostra Estratificada Proporcional mantendo a proporção de 'Genre'."""
    proporcoes = df["Genre"].value_counts(normalize=True)
    cotas = np.floor(proporcoes * n).astype(int)
    sobra = n - int(cotas.sum())
    if sobra > 0:
        maiores_residuos = ((proporcoes * n) - cotas).sort_values(ascending=False).index[:sobra]
        cotas.loc[maiores_residuos] += 1

    partes = [
        df[df["Genre"] == genero].sample(n=qtd, random_state=semente)
        for genero, qtd in cotas.items() if qtd > 0
    ]
    return pd.concat(partes, ignore_index=True)


def comparar_amostragens(
    df: pd.DataFrame,
    coluna: str = "Global_Sales",
    n: int = 60,
    semente: int = 42,
) -> None:
    """Compara AAS e Estratificada por Gênero contra Conveniência (topo do CSV)."""
    mu_pop = float(df[coluna].mean())
    sigma_pop = float(df[coluna].std(ddof=0))

    am_aas = df.sample(n=n, random_state=semente)[coluna]
    am_ae = amostrar_estratificada_genero(df, n=n, semente=semente)[coluna]
    am_conv = df.head(n)[coluna]  # Topo do ranking -> apenas blockbusters!

    linhas = [
        ("População Completa (μ)", "Referência", len(df), mu_pop, sigma_pop, 0.0),
        ("1. Aleatória Simples (AAS)", "Probabilística", n, float(am_aas.mean()), float(am_aas.std(ddof=1)), abs(float(am_aas.mean()) - mu_pop)),
        ("2. Estratificada por Gênero (AE)", "Probabilística", n, float(am_ae.mean()), float(am_ae.std(ddof=1)), abs(float(am_ae.mean()) - mu_pop)),
        ("3. Conveniência (Top 60 do CSV)", "Não Probabilística", n, float(am_conv.mean()), float(am_conv.std(ddof=1)), abs(float(am_conv.mean()) - mu_pop)),
    ]
    tabela = pd.DataFrame(linhas, columns=["Método", "Tipo", "Tamanho", "Média (x̄ ou μ)", "Desvio (S ou σ)", "Erro |x̄ - μ|"])
    display(tabela.set_index("Método").round(4))

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    df_plot = tabela.iloc[1:]
    cores = ["#2563eb" if tp == "Probabilística" else "#dc2626" for tp in df_plot["Tipo"]]
    barras = ax.barh(df_plot["Método"], df_plot["Média (x̄ ou μ)"], color=cores, alpha=0.88, height=0.55)
    ax.axvline(mu_pop, color="#16a34a", linestyle="--", lw=2.2, label=f"Média Real μ = {fmt_br(mu_pop)} mi")
    ax.invert_yaxis()
    ax.set_xlabel("Média de Vendas Globais (milhões de cópias)")
    ax.set_title(f"Impacto da Técnica de Amostragem (n = {n}) — Viés da Conveniência vs. Sorteio Probabilístico")
    for b in barras:
        w = b.get_width()
        ax.text(w + 0.35, b.get_y() + b.get_height() / 2, f"{fmt_br(w, 3)} mi", va="center", fontweight="bold")
    ax.set_xlim(0, float(df_plot["Média (x̄ ou μ)"].max()) * 1.18)
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.show()


def calcular_intervalo_confianca(
    amostra: pd.Series,
    confianca: float = 0.95,
    mu_real: float | None = None,
    sigma_pop: float | None = None,
    unidade: str = "mi",
) -> None:
    """Calcula e exibe o Intervalo de Confiança com Correção de Bessel (ddof=1) e Tabela Z Unicaudal."""
    dados = amostra.dropna()
    n = len(dados)
    alpha = 1.0 - confianca
    x_barra = float(dados.mean())
    s_bessel = float(dados.std(ddof=1))

    sigma_conhecido = sigma_pop is not None
    modelo = escolher_modelo(n, sigma_conhecido)
    desvio = float(sigma_pop) if sigma_conhecido else s_bessel
    simb_desvio = r"\sigma(x)" if sigma_conhecido else r"S(x)"
    erro_padrao = desvio / math.sqrt(n)

    if modelo["dist"] == "Z":
        area_unicaudal = 0.5 - (alpha / 2.0)
        crit = float(norm.ppf(1.0 - alpha / 2.0))
        simb_crit = r"Z_{\alpha/2}"
        linha_crit = (
            rf"- **Tabela $Z$ Unicaudal:** $P(0 \le z_0 \le Z_{{\alpha/2}}) = 0{{,}}5 - \frac{{\alpha}}{{2}} = "
            rf"0{{,}}5 - {fmt_tex(alpha/2)} = \mathbf{{{fmt_tex(area_unicaudal)}}} \implies Z_{{\alpha/2}} = {fmt_tex(crit)}$"
        )
    else:
        gl = modelo["gl"]
        crit = float(t.ppf(1.0 - alpha / 2.0, df=gl))
        simb_crit = rf"t_{{\alpha/2,\,{gl}}}"
        linha_crit = rf"- **Tabela $t\text{{-Student}}$ ($gl = {gl}$):** $\alpha/2 = {fmt_tex(alpha/2)} \implies {simb_crit} = {fmt_tex(crit)}$"

    eps = crit * erro_padrao
    ic_inf, ic_sup = x_barra - eps, x_barra + eps

    check_txt = ""
    if mu_real is not None:
        capturou = ic_inf <= mu_real <= ic_sup
        icone = "✅ **SIM! O IC capturou a média populacional real**" if capturou else "⚠️ **NÃO capturou nesta amostra**"
        check_txt = rf"- **Conferência ($\mu = {fmt_tex(mu_real)}\text{{ {unidade}}}$):** {icone} (${fmt_tex(ic_inf)} \le {fmt_tex(mu_real)} \le {fmt_tex(ic_sup)}$)."

    md_ic = rf"""
### 📏 Intervalo de Confiança ({fmt_br(confianca*100, 0)}%) — Amostra $n = {n}$
- **Relação Fundamental:** $\text{{Confiança }} ({fmt_tex(confianca*100, 0)}\%) + \alpha ({fmt_tex(alpha*100, 0)}\%) = 100\%$
- **Estimadores Amostrais (Correção de Bessel `ddof=1`):** $\bar{{x}} = {fmt_tex(x_barra)}\text{{ {unidade}}}$, $\quad S(x) = \sqrt{{\frac{{\sum (x_i - \bar{{x}})^2}}{{n - 1}}}} = {fmt_tex(s_bessel)}\text{{ {unidade}}}$
- **Fluxograma:** {modelo["regra"]}
{linha_crit}
- **Margem de Erro ($\varepsilon$) e Intervalo ($\bar{{x}} \pm \varepsilon$):**
  $$\varepsilon = {simb_crit} \cdot \frac{{{simb_desvio}}}{{\sqrt{{n}}}} = {fmt_tex(crit)} \cdot \frac{{{fmt_tex(desvio)}}}{{\sqrt{{{n}}}}} = {fmt_tex(crit)} \cdot {fmt_tex(erro_padrao)} = \mathbf{{{fmt_tex(eps)}\text{{ {unidade}}}}}$$
  $$\bar{{x}} - \varepsilon \le \mu \le \bar{{x}} + \varepsilon \implies \mathbf{{{fmt_tex(ic_inf)} \le \mu \le {fmt_tex(ic_sup)}}} \text{{ ({unidade})}}$$
{check_txt}
"""
    display(Markdown(md_ic))


def calcular_tamanho_amostra(
    sigma_x: float,
    erro_max: float = 0.15,
    confianca: float = 0.95,
    unidade: str = "mi",
) -> None:
    """Calcula o tamanho mínimo da amostra arredondando sempre para cima (math.ceil)."""
    alpha = 1.0 - confianca
    area_unicaudal = 0.5 - (alpha / 2.0)
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    n_exato = (z_crit * sigma_x / erro_max) ** 2
    n_min = math.ceil(n_exato)

    md_n = rf"""
### 🎯 Tamanho Mínimo da Amostra ($n$) para $\varepsilon = {fmt_tex(erro_max, 2)}\text{{ {unidade}}}$ ({fmt_br(confianca*100, 0)}% de confiança)
- **Tabela $Z$ Unicaudal:** $0{{,}}5 - \alpha/2 = {fmt_tex(area_unicaudal)} \implies Z_{{\alpha/2}} = {fmt_tex(z_crit)} \quad | \quad \sigma(x) = {fmt_tex(sigma_x)}\text{{ {unidade}}}$
  $$n = \left( Z_{{\alpha/2}} \cdot \frac{{\sigma(x)}}{{\varepsilon}} \right)^2 = \left( {fmt_tex(z_crit)} \cdot \frac{{{fmt_tex(sigma_x)}}}{{{fmt_tex(erro_max, 2)}}} \right)^2 = {fmt_tex(n_exato)} \xrightarrow{{\texttt{{math.ceil}}}} \mathbf{{n = {n_min}\text{{ jogos}}}}$$
"""
    display(Markdown(md_n))


# ==============================================================================
# PARTE 2: TESTES DE HIPÓTESES ("MITOS GAMER") E SIMULADOR AO VIVO
# ==============================================================================
def testar_hipotese(
    amostra: pd.Series,
    mu_0: float,
    alpha: float = 0.05,
    tipo: str = "bilateral",
    titulo: str = "Teste de Hipóteses",
    veredito_rejeita: str = "A afirmação alternativa (Ha) foi confirmada estatisticamente!",
    veredito_nao_rejeita: str = "Não há evidência estatística para rejeitar a média de referência (H0).",
    sigma_pop: float | None = None,
    unidade: str = "mi",
) -> None:
    """Executa o Teste de Hipóteses enxuto nas 6 etapas da disciplina e plota a curva Z ou t."""
    tipo = tipo.lower().strip()
    dados = amostra.dropna()
    n = len(dados)
    x_barra = float(dados.mean())
    s_bessel = float(dados.std(ddof=1))

    sigma_conhecido = sigma_pop is not None
    modelo = escolher_modelo(n, sigma_conhecido)
    dist, gl = modelo["dist"], modelo["gl"]
    desvio = float(sigma_pop) if sigma_conhecido else s_bessel
    simb_desvio = r"\sigma(x)" if sigma_conhecido else r"S(x)"
    simb_stat = "Z" if dist == "Z" else "t"
    nome_curva = "Normal Padrão (Z)" if dist == "Z" else f"t-Student (gl = {gl})"

    # 1. Hipóteses e cauda
    if tipo == "bilateral":
        hip_tex = rf"H_0: \mu = {fmt_tex(mu_0)} \quad \text{{vs.}} \quad H_a: \mu \neq {fmt_tex(mu_0)} \quad (\text{{Bilateral, }}\alpha/2 = {fmt_tex(alpha/2)}\text{{ por cauda}})"
        cauda = alpha / 2.0
    elif tipo == "esquerda":
        hip_tex = rf"H_0: \mu \ge {fmt_tex(mu_0)} \quad \text{{vs.}} \quad H_a: \mu < {fmt_tex(mu_0)} \quad (\text{{Unilateral à Esquerda, }}\alpha = {fmt_tex(alpha)})"
        cauda = alpha
    elif tipo == "direita":
        hip_tex = rf"H_0: \mu \le {fmt_tex(mu_0)} \quad \text{{vs.}} \quad H_a: \mu > {fmt_tex(mu_0)} \quad (\text{{Unilateral à Direita, }}\alpha = {fmt_tex(alpha)})"
        cauda = alpha
    else:
        raise ValueError("Use tipo='bilateral', 'esquerda' ou 'direita'.")

    # 3. Valor crítico e tabela
    crit = float(norm.ppf(1.0 - cauda)) if dist == "Z" else float(t.ppf(1.0 - cauda, df=gl))
    if dist == "Z":
        area_tab = 0.5 - cauda
        form_tab = r"0{,}5 - \alpha/2" if tipo == "bilateral" else r"0{,}5 - \alpha"
        info_tab = rf"Tabela $Z$ Unicaudal (${form_tab} = {fmt_tex(area_tab)}$)"
    else:
        info_tab = rf"Tabela $t\text{{-Student}}$ ($gl = {gl}$, cauda $= {fmt_tex(cauda)}$)"

    # 4. Estatística calculada
    erro_padrao = desvio / math.sqrt(n)
    stat_calc = (x_barra - mu_0) / erro_padrao
    cdf_val = float(norm.cdf(stat_calc)) if dist == "Z" else float(t.cdf(stat_calc, df=gl))

    if tipo == "bilateral":
        rc_tex = rf"R_c = (-\infty;\,-{fmt_tex(crit)}] \cup [+{fmt_tex(crit)};\,+\infty)"
        rnc_tex = rf"R_{{NC}} = (-{fmt_tex(crit)};\,+{fmt_tex(crit)})"
        rejeita = abs(stat_calc) >= crit
        p_val = 2.0 * min(cdf_val, 1.0 - cdf_val)
    elif tipo == "esquerda":
        rc_tex = rf"R_c = (-\infty;\,-{fmt_tex(crit)}]"
        rnc_tex = rf"R_{{NC}} = (-{fmt_tex(crit)};\,+\infty)"
        rejeita = stat_calc <= -crit
        p_val = cdf_val
    else:
        rc_tex = rf"R_c = [+{fmt_tex(crit)};\,+\infty)"
        rnc_tex = rf"R_{{NC}} = (-\infty;\,+{fmt_tex(crit)})"
        rejeita = stat_calc >= crit
        p_val = 1.0 - cdf_val

    if rejeita:
        decisao_tex = rf"{simb_stat}_{{calc}} = {fmt_tex(stat_calc)} \in R_c \implies \mathbf{{REJEITAMOS\; H_0}}"
        conclusao = f"✅ **Conclusão:** {veredito_rejeita}"
        erro_txt = f"⚠️ **Risco associado:** **Erro Tipo I ($\\alpha = {fmt_br(alpha*100, 1)}\\%$)** — rejeitar $H_0$ caso ela fosse verdadeira na população."
    else:
        decisao_tex = rf"{simb_stat}_{{calc}} = {fmt_tex(stat_calc)} \in R_{{NC}} \implies \mathbf{{NÃO\; REJEITAMOS\; H_0}}"
        conclusao = f"⚖️ **Conclusão:** {veredito_nao_rejeita}"
        erro_txt = "⚠️ **Risco associado:** **Erro Tipo II ($\\beta$)** — não rejeitar $H_0$ caso exista uma diferença real na população."

    resumo_md = rf"""
### 🧪 {titulo}
1. **Hipóteses:** ${hip_tex}$
2. **Amostra e Modelo ($1 - \alpha = {fmt_tex((1-alpha)*100, 0)}\%$):** $n = {n}$, $\bar{{x}} = {fmt_tex(x_barra)}\text{{ {unidade}}}$, ${simb_desvio} = {fmt_tex(desvio)}\text{{ {unidade}}}$ — {modelo["regra"]}
3. **Valor Crítico e Regiões ({info_tab}):** ${simb_stat}_c = {fmt_tex(crit)} \;\; \Rightarrow \;\; {rc_tex} \;\; | \;\; {rnc_tex}$
4. **Estatística Calculada:**
   $${simb_stat}_{{calc}} = \frac{{\bar{{x}} - \mu_0}}{{\frac{{{simb_desvio}}}{{\sqrt{{n}}}}}} = \frac{{{fmt_tex(x_barra)} - {fmt_tex(mu_0)}}}{{\frac{{{fmt_tex(desvio)}}}{{\sqrt{{{n}}}}}}} = \frac{{{fmt_tex(x_barra - mu_0)}}}{{{fmt_tex(erro_padrao)}}} = \mathbf{{{fmt_tex(stat_calc)}}}$$
5. **Decisão Formal (Região Crítica):** $${decisao_tex}$$
   *(Info complementar: $\text{{p-valor}} = {fmt_tex(p_val)}$ — decisão tomada exclusivamente por $R_c / R_{{NC}}$).*
6. **Interpretação:** {conclusao}
   - {erro_txt}
"""
    display(Markdown(resumo_md))

    # Gráfico limpo da curva Normal Z ou t-Student
    lim_x = max(4.2, abs(stat_calc) + 0.9, crit + 1.2)
    x = np.linspace(-lim_x, lim_x, 1000)
    y = norm.pdf(x) if dist == "Z" else t.pdf(x, df=gl)
    y_max = float(np.max(y))

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(x, y, color="#111827", lw=2.3, label=f"Curva {nome_curva}")
    caixa_crit = {"boxstyle": "round,pad=0.22", "fc": "#fff1f2", "ec": "#fecdd3", "alpha": 0.94}

    if tipo == "bilateral":
        ax.fill_between(x, y, where=(x >= -crit) & (x <= crit), color="#86efac", alpha=0.45, label=f"R_NC (1 - α = {fmt_br((1-alpha)*100, 0)}%)")
        ax.fill_between(x, y, where=(x <= -crit), color="#f87171", alpha=0.65, label=f"R_c (α/2 = {fmt_br(cauda*100, 2)}% por cauda)")
        ax.fill_between(x, y, where=(x >= crit), color="#f87171", alpha=0.65)
        ax.axvline(-crit, color="#b91c1c", ls="--", lw=1.8)
        ax.axvline(crit, color="#b91c1c", ls="--", lw=1.8)
        ax.text(-crit - 0.08, y_max * 0.54, f"-{simb_stat}c = -{fmt_br(crit)}", color="#991b1b", fontweight="bold", ha="right", bbox=caixa_crit)
        ax.text(crit + 0.08, y_max * 0.54, f"+{simb_stat}c = +{fmt_br(crit)}", color="#991b1b", fontweight="bold", ha="left", bbox=caixa_crit)
        ax.text(0, y_max * 0.35, "R_NC\n(Não Rejeita H₀)", ha="center", va="center", fontweight="bold", color="#166534")
        ax.text(-crit - 0.55, y_max * 0.10, "R_c", ha="center", fontweight="bold", color="#991b1b")
        ax.text(crit + 0.55, y_max * 0.10, "R_c", ha="center", fontweight="bold", color="#991b1b")
    elif tipo == "esquerda":
        ax.fill_between(x, y, where=(x >= -crit), color="#86efac", alpha=0.45, label=f"R_NC (1 - α = {fmt_br((1-alpha)*100, 0)}%)")
        ax.fill_between(x, y, where=(x <= -crit), color="#f87171", alpha=0.65, label=f"R_c à Esquerda (α = {fmt_br(alpha*100, 0)}%)")
        ax.axvline(-crit, color="#b91c1c", ls="--", lw=1.8)
        ax.text(-crit + 0.08, y_max * 0.54, f"-{simb_stat}c = -{fmt_br(crit)}", color="#991b1b", fontweight="bold", ha="left", bbox=caixa_crit)
        ax.text(0.45, y_max * 0.35, "R_NC\n(Não Rejeita H₀)", ha="center", va="center", fontweight="bold", color="#166534")
        ax.text(-crit - 0.55, y_max * 0.10, "R_c", ha="center", fontweight="bold", color="#991b1b")
    else:
        ax.fill_between(x, y, where=(x <= crit), color="#86efac", alpha=0.45, label=f"R_NC (1 - α = {fmt_br((1-alpha)*100, 0)}%)")
        ax.fill_between(x, y, where=(x >= crit), color="#f87171", alpha=0.65, label=f"R_c à Direita (α = {fmt_br(alpha*100, 0)}%)")
        ax.axvline(crit, color="#b91c1c", ls="--", lw=1.8)
        ax.text(crit - 0.08, y_max * 0.54, f"+{simb_stat}c = +{fmt_br(crit)}", color="#991b1b", fontweight="bold", ha="right", bbox=caixa_crit)
        ax.text(-0.45, y_max * 0.35, "R_NC\n(Não Rejeita H₀)", ha="center", va="center", fontweight="bold", color="#166534")
        ax.text(crit + 0.55, y_max * 0.10, "R_c", ha="center", fontweight="bold", color="#991b1b")

    cor_pt = "#7f1d1d" if rejeita else "#1d4ed8"
    y_pt = float(norm.pdf(stat_calc)) if dist == "Z" else float(t.pdf(stat_calc, df=gl))
    ax.axvline(stat_calc, color=cor_pt, lw=2.4, label=f"{simb_stat}_calc = {fmt_br(stat_calc)} ({'∈ R_c' if rejeita else '∈ R_NC'})")
    ax.scatter([stat_calc], [y_pt], color=cor_pt, s=85, zorder=6)

    desloc = -0.85 if stat_calc > 0 else 0.85
    ax.annotate(
        f"{simb_stat}_calc = {fmt_br(stat_calc)}",
        xy=(stat_calc, y_pt),
        xytext=(stat_calc + desloc, y_max * 0.76),
        arrowprops={"arrowstyle": "->", "color": cor_pt, "lw": 1.8},
        fontsize=10.5,
        fontweight="bold",
        color=cor_pt,
        bbox={"boxstyle": "round,pad=0.28", "fc": "#ffffff", "ec": cor_pt, "lw": 1.3},
        ha="center",
    )

    ax.set_ylim(0, y_max * 1.28)
    ax.set_xlabel(f"Escala Padronizada ({simb_stat})")
    ax.set_ylabel("Densidade f(x)")
    ax.set_title(f"{titulo} — {nome_curva}")
    ax.legend(loc="upper left", framealpha=0.93, fontsize=9.5)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def simular_ao_vivo(
    df: pd.DataFrame,
    mu_0: float,
    franquia: str | None = None,
    plataforma: str | None = None,
    genero: str | None = None,
    publicadora: str | None = None,
    coluna: str = "Global_Sales",
    n: int = 25,
    alpha: float = 0.05,
    tipo: str = "direita",
    semente: int = 42,
) -> None:
    """Filtra por franquia/console/gênero/publicadora, mostra amostra de jogos e roda o teste ao vivo."""
    sub = df.dropna(subset=[coluna]).copy()
    filtros = []

    if franquia:
        sub = sub[sub["Name"].str.contains(franquia, case=False, na=False)]
        filtros.append(f"Franquia='{franquia}'")
    if plataforma:
        sub = sub[sub["Platform"] == plataforma]
        filtros.append(f"Platform='{plataforma}'")
    if genero:
        sub = sub[sub["Genre"] == genero]
        filtros.append(f"Genre='{genero}'")
    if publicadora:
        sub = sub[sub["Publisher"] == publicadora]
        filtros.append(f"Publisher='{publicadora}'")

    desc_filtro = ", ".join(filtros) if filtros else "Catálogo Completo"
    if len(sub) < 2:
        display(Markdown(f"❌ **Filtro `{desc_filtro}` retornou menos de 2 jogos ({len(sub)}). Tente outro termo!**"))
        return

    n_efetivo = min(n, len(sub))
    if n_efetivo < n:
        display(Markdown(f"ℹ️ *O recorte `{desc_filtro}` possui **{len(sub)} jogos** no CSV; ajustando amostra automaticamente para `n = {n_efetivo}`.*"))

    amostra_df = sub.sample(n=n_efetivo, random_state=semente)
    display(Markdown(f"**🎮 Jogos sorteados na amostra (`{desc_filtro}`, $n = {n_efetivo}$ de {len(sub)} disponíveis — exibindo 5 exemplos):**"))
    exemplos = amostra_df[["Rank", "Name", "Platform", "Year", "Genre", "Publisher", coluna]].head(5).copy()
    if exemplos["Year"].notna().all():
        exemplos["Year"] = exemplos["Year"].astype(int)
    display(exemplos.set_index("Rank"))

    unidade = "ano" if coluna == "Year" else "mi"
    testar_hipotese(
        amostra=amostra_df[coluna],
        mu_0=mu_0,
        alpha=alpha,
        tipo=tipo,
        titulo=f"Ao Vivo: {desc_filtro} (n = {n_efetivo})",
        veredito_rejeita=f"Rejeitamos H₀! O recorte ({desc_filtro}) diverge significativamente de μ₀ = {fmt_br(mu_0)} {unidade}.",
        veredito_nao_rejeita=f"Não rejeitamos H₀! O recorte ({desc_filtro}) é estatisticamente compatível com μ₀ = {fmt_br(mu_0)} {unidade}.",
        unidade=unidade,
    )
