import numpy as np
import pandas as pd
from scipy.integrate import odeint
import streamlit as st
import plotly.express as px

def executar_influenza():
    """
    Simulação ajustável para Influenza spp. baseada em SIR/SEIR/SEIRS com:
      - Período latente (E) (SEIR)
      - Re-susceptibilidade (S <- R) (SEIRS)
      - Vacinação (fluxo S -> R)
      - Natalidade/mortalidade natural (μ)
      - Sazonalidade em β: beta(t) = beta0 * (1 + alpha * cos(2π(t - phi)/365))
    """

    st.header("Simulação - *Influenza spp.*")

    modelo = st.selectbox("Tipo de modelo", options=["SIR", "SEIR", "SEIRS (Re-susceptibilidade)"])


    with st.sidebar:
        st.subheader('Parâmetros Iniciais')

        # População e condições iniciais
        N = st.number_input("População total ($N$)", min_value=100, max_value=10_000_000_000, value=10_000)
        I0 = st.number_input("Infectados iniciais ($I_0$)", min_value=0, max_value=N, value=10)
        E0 = st.number_input("Expostos iniciais ($E_0$)", min_value=0, max_value=N, value=0)
        R0_init = st.number_input("Recuperados iniciais ($R_0$)", min_value=0, max_value=N, value=0)
        S0 = N - I0 - E0 - R0_init

        st.subheader('Parâmetros Dinâmicos')
        # Taxas epidemiológicas
        beta0 = st.slider(r"Taxa de transmissão básica ($\beta_0$)", 0.0, 2.0, 0.8, 0.01)
        alpha = st.slider(r"Amplitude sazonal ($\alpha$)", 0.0, 1.0, 0.2, 0.01)
        phi = st.number_input(r"Deslocamento sazonal (dias) $\phi$", min_value=0, max_value=365, value=0)
        sigma = st.slider("Taxa de progressão $E->I (σ)$ (1/latência dias)", 0.0, 1.0, 1/1.5, 0.01)  # latência ~1-2 dias
        gamma = st.slider(r"Taxa de recuperação ($\gamma$) (1/dias infecc.)", 0.0, 1.0, 1/3.0, 0.01)  # recuperação ~3 dias
        mu = st.slider("Taxa mortalidade/natalidade natural (μ) anual -> convert. diária", 0.0, 0.05, 1/(70*365), 1e-6)
        # Nota: mu fornecido em taxa diária (usuário pode ajustar) -> Padrão: ~1/(70*365)

        # Re-susceptibilidade
        waning_days = st.number_input("Dias médios até perda de imunidade (0 = permanente)", 0, 3650, 365)
        if waning_days == 0:
            nu = 0.0
        else:
            nu = 1.0 / waning_days

        # Vacinação
        vac_on = False
        if modelo in ["SEIRV (vacinação)", "SEIRS (Re-susceptibilidade)"]:
            vac_on = st.checkbox("Ativar vacinação (fluxo S -> R)", value=False)
        else:
            vac_on = st.checkbox("Ativar vacinação (fluxo S -> R)", value=False)
        if vac_on:
            v_rate = st.slider("Taxa de vacinação diária (fração da população suscetível)", 0.0, 1.0, 0.001, 0.0001)
            vac_efficacy = st.slider("Eficácia da vacina (0-1)", 0.0, 1.0, 0.6, 0.01)
        else:
            v_rate = 0.0
            vac_efficacy = 0.0

        # Escolhas de visualização
        dias = st.slider("Dias de simulação", 1, 3650, 365, 1)
        st.subheader("Curvas exibidas")
        mostrar_S = st.checkbox("Susceptíveis", value=True)
        mostrar_E = st.checkbox("Expostos", value=(modelo != "SIR"))
        mostrar_I = st.checkbox("Infectados", value=True)
        mostrar_R = st.checkbox("Recuperados", value=True)

    # Função beta sazonal
    def beta_t(t):
        return beta0 * (1.0 + alpha * np.cos(2.0 * np.pi * (t - phi) / 365.0))

    # Sistema de EDO's
    def modelo_influenza(y, t):
        if modelo == "SIR":
            S, I, R = y
            b = beta_t(t)
            dS = mu * N - b * S * I / N - mu * S - v_rate * vac_efficacy * S
            dI = b * S * I / N - (gamma + mu) * I
            dR = gamma * I - mu * R + v_rate * vac_efficacy * S
            return [dS, dI, dR]

        elif modelo == "SEIR":
            S, E, I, R = y
            b = beta_t(t)
            dS = mu * N - b * S * I / N - mu * S - v_rate * vac_efficacy * S
            dE = b * S * I / N - (sigma + mu) * E
            dI = sigma * E - (gamma + mu) * I
            dR = gamma * I - mu * R + v_rate * vac_efficacy * S
            return [dS, dE, dI, dR]

        elif modelo == "SEIRS (Re-susceptibilidade)":
            S, E, I, R = y
            b = beta_t(t)
            dS = mu * N - b * S * I / N - mu * S - v_rate * vac_efficacy * S + nu * R
            dE = b * S * I / N - (sigma + mu) * E
            dI = sigma * E - (gamma + mu) * I
            dR = gamma * I - mu * R - nu * R + v_rate * vac_efficacy * S
            return [dS, dE, dI, dR]

        else:
            raise ValueError("Modelo não implementado")

    # Condições iniciais vetoriais
    if modelo == "SIR":
        y0 = [S0, I0, R0_init]
    else:
        y0 = [S0, E0, I0, R0_init]

    # Tempo
    t = np.linspace(0, dias, dias + 1)

    # Resolve as EDO's
    resultado = odeint(modelo_influenza, y0, t)
    # Extração dos resultados
    if modelo == "SIR":
        S, I, R = resultado.T
        data = pd.DataFrame({"Dias": t, "Susceptíveis": S, "Infectados": I, "Recuperados": R})
    else:
        S, E, I, R = resultado.T
        data = pd.DataFrame({"Dias": t, "Susceptíveis": S, "Expostos": E, "Infectados": I, "Recuperados": R})

    # Filtrar colunas para plot baseado nas checkboxes
    plot_df = pd.DataFrame({"Dias": t})
    if mostrar_S and "Susceptíveis" in data:
        plot_df["Susceptíveis"] = data["Susceptíveis"]
    if mostrar_E and "Expostos" in data:
        plot_df["Expostos"] = data["Expostos"]
    if mostrar_I:
        plot_df["Infectados"] = data["Infectados"]
    if mostrar_R:
        plot_df["Recuperados"] = data["Recuperados"]

    # Transformação do DataFrame para "long"
    data_long = plot_df.melt(id_vars="Dias", var_name="Categoria", value_name="Número de Indivíduos")

    # Cores
    color_map = {
        "Susceptíveis": "blue",
        "Expostos": "orange",
        "Infectados": "red",
        "Recuperados": "green"
    }

    fig = px.line(data_long, x="Dias", y="Número de Indivíduos", color="Categoria",
                  title=f"Modelo {modelo}", template="plotly_white", color_discrete_map=color_map)

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(text=f"Modelo {modelo}", x=0.5, xanchor="center", font=dict(size=20, color="black")),
    )
    fig.update_xaxes(showline=True, linewidth=1.5, tickfont=dict(color="black"), title=dict(text="Dias", font=dict(color="black")))
    fig.update_yaxes(showline=True, linewidth=1.5, tickfont=dict(color="black"), title=dict(text="Número de Indivíduos", font=dict(color="black")))
    fig.update_layout(legend_title_text="Categoria")
    fig.update_legends(title=dict(text='Categoria', font=dict(color="black"))) # Título da legenda
    fig.update_legends(font=dict(color='black')) # Elementos da legenda

    st.plotly_chart(fig, use_container_width=True)

    # Estatísticas de pico
    if "Infectados" in data:
        pico = t[np.argmax(data["Infectados"].values)]
        max_infeccoes = np.max(data["Infectados"].values)
    else:
        pico = None
        max_infeccoes = None

    st.subheader("Pico da epidemia")
    if pico is not None:
        st.markdown(f"- **Dia do pico**: {int(pico)}  \n- **Número máximo de infectados simultâneos**: {int(max_infeccoes)}")
    else:
        st.markdown("Sem dados de infectados para calcular pico.")

    # Distribuição final
    st.subheader("Distribuição final da população")
    total_final = data.iloc[-1][[c for c in data.columns if c != "Dias"]].sum()
    cols = [c for c in data.columns if c != "Dias"]
    cols_display = st.columns(len(cols))
    for col_name, col in zip(cols, cols_display):
        val = data[col_name].values[-1]
        delta = val - (S0 if col_name == "Susceptíveis" else (I0 if col_name == "Infectados" else (E0 if col_name == "Expostos" else R0_init)))
        with col:
            st.metric(col_name, value=f"{int(val)}", delta=f"{int(delta)}")
            st.markdown(f"{col_name}: {(val/total_final)*100:.2f}%")

    # Exibe R0 básico aproximado (quando aplicável)
    # Para SEIR/SEIRS, R0 ≈ beta0 / gamma (se sigma >> gamma, suposições são ignoradas) — dá-se uma estimativa simples
    st.subheader(r"Estimativa simples de $R_0$")
    if gamma > 0:
        R0_simple = beta0 / gamma
        st.write(r"$R_0$ (estimado simplificado) = $\frac{\beta_0}{\gamma}$ =", f"{R0_simple:.2f}")
        st.caption("Observação: esta é uma estimativa simplificada. Modelos com latência e sazonalidade afetam R₀ efetivo.")
    else:
        st.write(r"\gamma = 0, não é possível estimar $R_0$.")
