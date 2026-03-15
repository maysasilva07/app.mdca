import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from fpdf import FPDF
from docx import Document
from docx.shared import Inches
import tempfile
import os

# Configuração de fonte
plt.rcParams['font.family'] = 'DejaVu Sans'

# Funções matemáticas (mantidas iguais)
def normalizar_critic(df, criterios_tipo):
    df_norm = pd.DataFrame(index=df.index)
    for col in df.columns:
        r_min = df[col].min()
        r_max = df[col].max()
        if r_max == r_min:
            df_norm[col] = 0.0
            continue
        if criterios_tipo[col] == 'max':
            df_norm[col] = (df[col] - r_min) / (r_max - r_min)
        else:
            df_norm[col] = (r_max - df[col]) / (r_max - r_min)
    return df_norm

def metodo_critic(df_norm):
    desvios = df_norm.std(ddof=1)
    correlacao = df_norm.corr()
    info_c = pd.Series(index=df_norm.columns, dtype=float)
    for col in df_norm.columns:
        soma_conflito = (1 - correlacao[col]).sum()
        info_c[col] = desvios[col] * soma_conflito
    pesos = info_c / info_c.sum()
    return pesos.to_dict(), desvios, correlacao, info_c

def funcao_preferencia_tipo_v(d, q, p):
    if d <= q:
        return 0.0
    elif d >= p:
        return 1.0
    else:
        return (d - q) / (p - q)

def promethee(df, pesos, criterios_tipo, q_limites, p_limites):
    n = len(df)
    alternativas = list(df.index)
    criterios = list(df.columns)
    pref_agregada = pd.DataFrame(0.0, index=alternativas, columns=alternativas)
    for a in alternativas:
        for b in alternativas:
            if a == b:
                continue
            soma_pi = 0.0
            for crit in criterios:
                if criterios_tipo[crit] == 'max':
                    d = df.loc[a, crit] - df.loc[b, crit]
                else:
                    d = df.loc[b, crit] - df.loc[a, crit]
                q = q_limites[crit]
                p = p_limites[crit]
                if p == q:
                    pref = 1.0 if d > q else 0.0
                else:
                    pref = funcao_preferencia_tipo_v(d, q, p)
                soma_pi += pesos[crit] * pref
            pref_agregada.loc[a, b] = soma_pi
    phi_mais = pref_agregada.sum(axis=1) / (n - 1)
    phi_menos = pref_agregada.sum(axis=0) / (n - 1)
    phi_liquido = phi_mais - phi_menos
    ranking = phi_liquido.sort_values(ascending=False).rank(ascending=False, method='min').astype(int)
    return pref_agregada, phi_mais, phi_menos, phi_liquido, ranking

# Funções de gráfico
def gerar_grafico_pesos(pesos):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pesos.keys(), pesos.values(), color='royalblue')
    ax.set_title('Peso dos Critérios (CRITIC)')
    ax.set_ylabel('Peso (%)')
    plt.xticks(rotation=45)
    return fig

def gerar_grafico_fluxos(phi_mais, phi_menos, phi_liquido):
    df_plot = pd.DataFrame({'Fluxo+': phi_mais, 'Fluxo-': phi_menos, 'Fluxo Líquido': phi_liquido})
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
    ax.set_title('Fluxos de Superação - PROMETHEE')
    ax.set_ylabel('Valores dos Fluxos')
    plt.xticks(rotation=0)
    ax.axhline(0, color='black', linewidth=0.8)
    return fig

def gerar_grafo_sobreclassificacao(phi_mais, phi_menos):
    G = nx.DiGraph()
    alt_list = list(phi_mais.index)
    G.add_nodes_from(alt_list)
    for a in alt_list:
        for b in alt_list:
            if a == b:
                continue
            phi_mais_a = phi_mais[a]
            phi_mais_b = phi_mais[b]
            phi_menos_a = phi_menos[a]
            phi_menos_b = phi_menos[b]
            cond1 = (phi_mais_a > phi_mais_b) and (phi_menos_a < phi_menos_b)
            cond2 = (phi_mais_a == phi_mais_b) and (phi_menos_a < phi_menos_b)
            cond3 = (phi_mais_a > phi_mais_b) and (phi_menos_a == phi_menos_b)
            if cond1 or cond2 or cond3:
                G.add_edge(a, b)
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3500, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=25, edge_color='gray', connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_family='DejaVu Sans', ax=ax)
    ax.set_title("Grafo de Sobreclassificação (PROMETHEE I)", fontsize=16, pad=20)
    ax.axis('off')
    plt.tight_layout()
    return fig

# Funções de relatório (PDF e DOCX) - omitidas por brevidade, mas devem ser mantidas iguais
# ... (incluir as funções gerar_relatorio_pdf e gerar_relatorio_docx como antes)

# Função para resetar estado
def reset_app():
    keys = ['dados_entrada', 'criterios_tipos', 'criterios_qs', 'criterios_ps', 
            'metodo_peso', 'pesos_manuais', 'resultados']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

# Interface principal
def main():
    st.set_page_config(page_title="MCDA - CRITIC + PROMETHEE", layout="wide")
    st.title("📊 Análise Multicritério (CRITIC + PROMETHEE I/II)")

    # Botão Nova Análise no topo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🔄 Nova Análise", use_container_width=True):
            reset_app()
            st.rerun()

    # Inicialização de estados
    if 'dados_entrada' not in st.session_state:
        st.session_state.dados_entrada = None
    if 'criterios_tipos' not in st.session_state:
        st.session_state.criterios_tipos = {}
    if 'criterios_qs' not in st.session_state:
        st.session_state.criterios_qs = {}
    if 'criterios_ps' not in st.session_state:
        st.session_state.criterios_ps = {}
    if 'metodo_peso' not in st.session_state:
        st.session_state.metodo_peso = "Calcular automaticamente (CRITIC)"
    if 'pesos_manuais' not in st.session_state:
        st.session_state.pesos_manuais = None
    if 'resultados' not in st.session_state:
        st.session_state.resultados = {}

    # Seção 1: Entrada de dados
    st.header("1. Entrada de Dados")
    opcao = st.radio("Escolha a forma de entrada:", ["Upload de arquivo (Excel/CSV)", "Inserir manualmente"])
    if opcao == "Upload de arquivo (Excel/CSV)":
        arquivo = st.file_uploader("Faça upload da planilha", type=['xlsx', 'csv'])
        if arquivo is not None:
            try:
                if arquivo.name.endswith('.csv'):
                    df = pd.read_csv(arquivo, index_col=0, encoding='utf-8')
                else:
                    df = pd.read_excel(arquivo, index_col=0)
                st.session_state.dados_entrada = df
                st.success("Arquivo carregado!")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Erro: {e}")
    else:
        st.subheader("Defina as dimensões")
        cols_dim = st.columns(2)
        num_alt = cols_dim[0].number_input("Número de Alternativas", min_value=2, value=3, step=1)
        num_crit = cols_dim[1].number_input("Número de Critérios", min_value=1, value=3, step=1)
        if num_alt and num_crit:
            nomes_alt = [st.sidebar.text_input(f"Alt {i+1}", value=f"A{i+1}") for i in range(num_alt)]
            nomes_crit = [st.sidebar.text_input(f"Crit {j+1}", value=f"C{j+1}") for j in range(num_crit)]
            dados = []
            st.write("**Preencha as avaliações:**")
            for i, alt in enumerate(nomes_alt):
                cols_val = st.columns(num_crit)
                linha = [cols_val[j].number_input(f"{alt} - {crit}", value=0.0, step=0.1, key=f"v_{i}_{j}") for j, crit in enumerate(nomes_crit)]
                dados.append(linha)
            if st.button("Carregar matriz manual"):
                st.session_state.dados_entrada = pd.DataFrame(dados, index=nomes_alt, columns=nomes_crit)
                st.success("Matriz carregada!")
                st.dataframe(st.session_state.dados_entrada)

    # Seção 2: Parâmetros dos critérios
    if st.session_state.dados_entrada is not None:
        st.header("2. Parâmetros dos Critérios (PROMETHEE)")
        st.markdown("Defina o tipo (max/min) e os limiares q e p para a função de preferência Tipo V.")
        df = st.session_state.dados_entrada
        criterios = df.columns.tolist()

        # Inicializar estados para cada critério
        for crit in criterios:
            if crit not in st.session_state.criterios_tipos:
                st.session_state.criterios_tipos[crit] = "max"
            if crit not in st.session_state.criterios_qs:
                st.session_state.criterios_qs[crit] = 0.0
            if crit not in st.session_state.criterios_ps:
                st.session_state.criterios_ps[crit] = 0.0

        # Exibir inputs para cada critério
        valid_limiares = True
        for crit in criterios:
            st.write(f"**{crit}**")
            cols = st.columns(3)
            with cols[0]:
                tipo = st.selectbox("Objetivo", ["max", "min"], 
                                    index=0 if st.session_state.criterios_tipos[crit]=="max" else 1,
                                    key=f"tipo_{crit}")
                st.session_state.criterios_tipos[crit] = tipo
            with cols[1]:
                q = st.number_input("q (indiferença)", min_value=0.0, value=st.session_state.criterios_qs[crit], step=0.01, key=f"q_{crit}")
                st.session_state.criterios_qs[crit] = q
            with cols[2]:
                p = st.number_input("p (preferência)", min_value=0.0, value=st.session_state.criterios_ps[crit], step=0.01, key=f"p_{crit}")
                st.session_state.criterios_ps[crit] = p
            if p < q:
                st.error(f"p deve ser >= q para {crit}")
                valid_limiares = False

        # Método de pesos
        st.subheader("Método de Definição dos Pesos")
        metodo_peso = st.radio("Escolha:", ["Calcular automaticamente (CRITIC)", "Inserir manualmente"],
                               index=0 if st.session_state.metodo_peso == "Calcular automaticamente (CRITIC)" else 1)
        st.session_state.metodo_peso = metodo_peso

        if metodo_peso == "Inserir manualmente":
            st.markdown("Digite os pesos (valores entre 0 e 1). A soma será normalizada.")
            pesos_manuais = {}
            total = 0.0
            cols_peso = st.columns(len(criterios))
            for i, crit in enumerate(criterios):
                with cols_peso[i]:
                    default = 1.0 / len(criterios)
                    key = f"peso_manual_{crit}"
                    if key not in st.session_state:
                        st.session_state[key] = default
                    peso = st.number_input(f"{crit}", min_value=0.0, max_value=1.0, value=st.session_state[key], step=0.05, key=key)
                    pesos_manuais[crit] = peso
                    total += peso
            st.write(f"Soma atual: {total:.4f}")
            if abs(total - 1.0) > 0.01:
                st.warning("A soma não é 1. Será normalizada.")
            if st.button("Confirmar pesos manuais"):
                if total > 0:
                    st.session_state.pesos_manuais = {k: v/total for k, v in pesos_manuais.items()}
                    st.success("Pesos normalizados e salvos!")
                else:
                    st.error("Soma zero.")

        # Botão de execução
        executar_habilitado = valid_limiares
        if metodo_peso == "Inserir manualmente" and st.session_state.pesos_manuais is None:
            executar_habilitado = False
            st.warning("Defina os pesos manuais antes de executar.")

        if st.button("▶️ Rodar CRITIC + PROMETHEE", disabled=not executar_habilitado):
            df = st.session_state.dados_entrada
            tipos = st.session_state.criterios_tipos
            qs = st.session_state.criterios_qs
            ps = st.session_state.criterios_ps
            with st.spinner("Calculando..."):
                df_norm = normalizar_critic(df, tipos)
                if metodo_peso == "Calcular automaticamente (CRITIC)":
                    pesos, desvios, correl, info_c = metodo_critic(df_norm)
                    st.subheader("Pesos CRITIC")
                    col1, col2 = st.columns(2)
                    col1.dataframe(info_c.to_frame(name="Cj"))
                    col2.dataframe(pd.Series(pesos, name="Peso"))
                    st.pyplot(gerar_grafico_pesos(pesos))
                else:
                    pesos = st.session_state.pesos_manuais
                    correl = pd.DataFrame()
                    st.subheader("Pesos Manuais")
                    st.json(pesos)
                    st.pyplot(gerar_grafico_pesos(pesos))

                pref_matrix, phi_mais, phi_menos, phi_liquido, ranking = promethee(df, pesos, tipos, qs, ps)
                st.subheader("Resultados PROMETHEE")
                st.write("Matriz de Preferência Global (Π):")
                st.dataframe(pref_matrix)
                res_df = pd.DataFrame({
                    'Φ+': phi_mais, 'Φ-': phi_menos, 'Φ Líquido': phi_liquido,
                    'Ranking': ranking
                }).sort_values(by='Ranking')
                st.dataframe(res_df)
                colg1, colg2 = st.columns(2)
                with colg1:
                    st.pyplot(gerar_grafico_fluxos(phi_mais, phi_menos, phi_liquido))
                with colg2:
                    st.pyplot(gerar_grafo_sobreclassificacao(phi_mais, phi_menos))

                # Salvar resultados
                st.session_state.resultados = {
                    'entrada': df, 'norm': df_norm, 'correl': correl,
                    'pref_matrix': pref_matrix, 'pesos': pesos,
                    'phi_mais': phi_mais, 'phi_menos': phi_menos,
                    'phi_liquido': phi_liquido, 'ranking': ranking,
                    'fig_pesos': gerar_grafico_pesos(pesos),
                    'fig_fluxos': gerar_grafico_fluxos(phi_mais, phi_menos, phi_liquido),
                    'fig_grafo': gerar_grafo_sobreclassificacao(phi_mais, phi_menos),
                    'q_limites': qs, 'p_limites': ps, 'tipos': tipos
                }
                st.success("Análise concluída!")

    # Seção 4: Relatório
    if st.session_state.resultados:
        st.header("4. Gerar Relatório")
        formato = st.selectbox("Formato", ["PDF", "DOCX"])
        if st.button("📄 Gerar e baixar relatório"):
            # Funções de relatório (devem estar definidas acima)
            # ... (omitido para brevidade, mas deve ser incluído)
            pass

if __name__ == "__main__":
    main()
