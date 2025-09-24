import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import os

# ==============================
# 1. Carregar dados
# ==============================
@st.cache_data
def load_data():
    csv_path = "imdb-top-rated-movies-user-rated.csv"
    if not os.path.exists(csv_path):
        st.error(f"Arquivo CSV não encontrado em: {os.path.abspath(csv_path)}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df

df = load_data()

st.title("🎬 Sistema de Recomendação de Filmes (RBC)")
st.write("Base de dados carregada:", df.shape[0], "filmes")

# Mostrar primeiras linhas para debug
if st.checkbox("Mostrar dados brutos"):
    st.dataframe(df.head())

# ==============================
# 2. Pré-processamento
# ==============================
if df.empty:
    st.warning("⚠️ Base de dados vazia. Importe o CSV correto para prosseguir.")
else:
    # Limpeza de votos
    def clean_votes(vote_str):
        if pd.isna(vote_str):
            return 0
        try:
            if 'K' in str(vote_str):
                return float(str(vote_str).replace('K', '')) * 1000
            else:
                return float(vote_str)
        except:
            return 0

    df['Votes_Clean'] = df['Votes'].apply(clean_votes) if 'Votes' in df.columns else 0
    df['Year_Extracted'] = df['Title'].str.extract(r'\((\d{4})\)').fillna(0)
    df['Year_Extracted'] = pd.to_numeric(df['Year_Extracted'], errors='coerce').fillna(0).astype(int)
    
    # Features numéricas (só as que existem)
    features = []
    for col in ['IMDb Rating', 'Meta Score', 'Votes_Clean', 'Year_Extracted']:
        if col in df.columns:
            features.append(col)
    df_proc = df[features].fillna(0)
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_proc)
    
    st.success("✅ Dados processados com sucesso!")
    st.write("Estatísticas das features:")
    st.write(df_proc.describe())

# ==============================
# 3. Função de recomendação (RBC)
# ==============================
def recomendar_filmes(nome_filme, top_k=5):
    try:
        mask = df['Title'].str.lower().str.contains(nome_filme.lower(), na=False)
        if not mask.any():
            return None, "Filme não encontrado na base de dados."
        
        idx = df[mask].index[0]
        
        # Colunas disponíveis para exibir
        colunas_disp = ['Title', 'Year_Extracted', 'IMDb Rating']
        if 'Genre' in df.columns:
            colunas_disp.append('Genre')
        
        filme_original = df.iloc[idx][colunas_disp]
        sim_scores = cosine_similarity([df_scaled[idx]], df_scaled)[0]
        indices = np.argsort(sim_scores)[::-1][1:top_k+1]  # pula o próprio filme
        recomendados = df.iloc[indices][colunas_disp]
        
        return filme_original, recomendados
    except Exception as e:
        return None, f"Erro na recomendação: {e}"

# ==============================
# 4. Interface Streamlit
# ==============================
if not df.empty:
    st.subheader("🔍 Buscar Recomendações")
    
    filmes_disponiveis = df['Title'].sort_values().tolist()
    filme_selecionado = st.selectbox("Selecione um filme:", filmes_disponiveis, index=0)
    top_k = st.slider("Número de recomendações:", min_value=3, max_value=10, value=5)
    
    if st.button("Buscar Recomendações"):
        if filme_selecionado:
            with st.spinner("Buscando recomendações..."):
                filme_original, recomendacoes = recomendar_filmes(filme_selecionado, top_k)
                
                if filme_original is not None and recomendacoes is not None:
                    st.success(f"🎯 Filme base: **{filme_original['Title']}** ({filme_original['Year_Extracted']})")
                    st.write(f"⭐ Rating: {filme_original['IMDb Rating']}" + (f" | 🏷️ Gênero: {filme_original['Genre']}" if 'Genre' in filme_original else ""))
                    
                    st.subheader("🎬 Filmes Recomendados:")
                    for i, (_, filme) in enumerate(recomendacoes.iterrows(), 1):
                        col1, col2 = st.columns([3,1])
                        with col1:
                            st.write(f"**{i}. {filme['Title']}** ({filme['Year_Extracted']})")
                            if 'Genre' in filme:
                                st.caption(f"Gênero: {filme['Genre']}")
                        with col2:
                            st.metric("Rating", filme['IMDb Rating'])
                        st.divider()
                else:
                    st.error(recomendacoes)

# ==============================
# 5. Sidebar e informações
# ==============================
if not df.empty:
    st.sidebar.header("ℹ️ Sobre o Sistema")
    st.sidebar.write("""
    Sistema baseado em **Raciocínio Baseado em Casos (RBC)**.
    Similaridade baseada em:
    - ⭐ Avaliação IMDb
    - 🏆 Meta Score
    - 👥 Número de votos
    - 📅 Ano de lançamento
    """)
    
    st.sidebar.header("📊 Estatísticas")
    st.sidebar.metric("Total de Filmes", df.shape[0])
    if 'IMDb Rating' in df.columns:
        st.sidebar.metric("Rating Médio", f"{df['IMDb Rating'].mean():.1f}")
    
    try:
        votes_max = df['Votes_Clean'].max() if 'Votes_Clean' in df.columns else 0
        if votes_max >= 1_000_000:
            votes_text = f"{votes_max/1_000_000:.1f}M votos"
        elif votes_max >= 1_000:
            votes_text = f"{votes_max/1_000:.0f}K votos"
        else:
            votes_text = f"{votes_max:.0f} votos"
        st.sidebar.metric("Filme Mais Votado", votes_text)
    except:
        st.sidebar.metric("Filme Mais Votado", "N/A")
    
    # Explorar por gênero (só se coluna existir)
    if 'Genre' in df.columns:
        st.sidebar.header("🎭 Explorar por Gênero")
        todos_generos = []
        for generos in df['Genre'].dropna():
            generos_split = [g.strip() for g in generos.split(',')]
            todos_generos.extend(generos_split)
        generos_disponiveis = sorted(set(todos_generos))
        genero_selecionado = st.sidebar.selectbox("Selecione um gênero:", generos_disponiveis)
        if genero_selecionado:
            filmes_genero = df[df['Genre'].str.contains(genero_selecionado, na=False)].nlargest(5, 'IMDb Rating')
            st.sidebar.write(f"**Top 5 {genero_selecionado}:**")
            for _, filme in filmes_genero.iterrows():
                st.sidebar.write(f"• {filme['Title']} ({filme['IMDb Rating']}⭐)")
