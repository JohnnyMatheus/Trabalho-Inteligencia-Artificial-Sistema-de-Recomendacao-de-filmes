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
    if 'Year' in df.columns:
        df['Year_Extracted'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
    else:
        df['Year_Extracted'] = 0
    
    # Features numéricas
    features = []
    # Adiciona campos numéricos relevantes
    for col in ['IMDb Rating', 'Meta Score', 'Votes_Clean', 'Year_Extracted']:
        if col in df.columns or col in ['Votes_Clean', 'Year_Extracted']:
            features.append(col)

    # Adiciona campos categóricos relevantes (Diretor, Stars)
    if 'Director' in df.columns:
        df['Director_clean'] = df['Director'].fillna('').str.lower().str.strip()
        directores_unicos = df['Director_clean'].unique()
        for d in directores_unicos:
            if d:
                features.append(f'director_{d}')
                df[f'director_{d}'] = (df['Director_clean'] == d).astype(int)
    if 'Stars' in df.columns:
        # Extrai atores principais
        def split_stars(s):
            if pd.isna(s): return []
            return [a.strip().lower() for a in str(s).replace('"', '').split(',') if a.strip()]
        all_stars = set()
        for s in df['Stars']:
            all_stars.update(split_stars(s))
        # Cria todas as colunas de atores de uma vez só
        stars_matrix = []
        for s in df['Stars']:
            stars_set = set(split_stars(s))
            stars_matrix.append([int(star in stars_set) for star in all_stars])
        stars_df = pd.DataFrame(stars_matrix, columns=[f'star_{star}' for star in all_stars], index=df.index)
        df = pd.concat([df, stars_df], axis=1)
        features.extend(stars_df.columns.tolist())
    
    # ==============================
    # One-hot encoding de gêneros
    # ==============================
    if 'Genre' in df.columns:
        generos_unicos = set()
        for g in df['Genre'].dropna():
            generos_unicos.update([x.strip() for x in g.split(',')])
        generos_unicos = sorted(list(generos_unicos))
        
        for genero in generos_unicos:
            df[genero] = df['Genre'].apply(lambda x: 1 if pd.notna(x) and genero in x else 0)
            features.append(genero)
    
    # Normalizar
    df_proc = df[features].fillna(0)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_proc)
    
    st.success("✅ Dados processados com sucesso!")
    st.write("Estatísticas das features numéricas:")
    st.write(df_proc.describe())

# ==============================
# 3. Função de recomendação (RBC)
# ==============================
def recomendar_filmes(nome_filme, top_k=5):
    try:
        mask = df['Title'].str.lower().str.contains(nome_filme.lower(), na=False)
        if not mask.any():
            return None, "Filme não encontrado na base de dados.", None, None, None

        idx = df[mask].index[0]
        def normaliza_generos(s):
            if pd.isna(s) or str(s).strip() == '':
                return set()
            return set([g.strip().lower().replace('-', '').replace(' ', '') for g in str(s).replace('"', '').split(',') if g.strip()])

        generos_base = set()
        if 'Genre' in df.columns and idx in df.index:
            val = df.at[idx, 'Genre'] if 'Genre' in df.columns else ''
            generos_base = normaliza_generos(val)

        diretor_base = ''
        if 'Director_clean' in df.columns and idx in df.index:
            diretor_base = df.at[idx, 'Director_clean']
        stars_base = set()
        if 'Stars' in df.columns and idx in df.index:
            def split_stars(s):
                if pd.isna(s): return set()
                return set([a.strip().lower() for a in str(s).replace('"', '').split(',') if a.strip()])
            stars_base = split_stars(df.at[idx, 'Stars'])

        tags_base = set()
        if 'Tags' in df.columns and idx in df.index:
            val = df.at[idx, 'Tags'] if 'Tags' in df.columns else ''
            if pd.notna(val) and str(val).strip() != '':
                tags_base = set([t.strip().lower() for t in str(val).split(',') if t.strip()])

        # Filtrar apenas filmes com pelo menos 1 gênero em comum (obrigatório)
        def tem_genero_em_comum(x):
            generos = normaliza_generos(x)
            return len(generos & generos_base) > 0

        if 'Genre' in df.columns:
            mask_genero = df['Genre'].apply(tem_genero_em_comum)
        else:
            mask_genero = pd.Series([False]*len(df), index=df.index)
        indices_validos = df[mask_genero].index.tolist()
        if idx in indices_validos:
            indices_validos.remove(idx)  # Remove o próprio filme
        genero_relaxado = False
        if not indices_validos:
            # Relaxa a restrição: recomenda os mais similares no geral (exceto o próprio)
            genero_relaxado = True
            indices_validos = list(df.index)
            if idx in indices_validos:
                indices_validos.remove(idx)
        # Similaridade só entre os válidos
        sim_scores = cosine_similarity([df_scaled[idx]], df_scaled[indices_validos])[0]
        # Ordenar por similaridade
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        indices_recomendados = [indices_validos[i] for i in top_indices]

        colunas_disp = ['Title', 'Year_Extracted', 'IMDb Rating']
        if 'Genre' in df.columns:
            colunas_disp.append('Genre')
        if 'Tags' in df.columns:
            colunas_disp.append('Tags')

        recomendados = df.loc[indices_recomendados, colunas_disp]
        filme_original = df.loc[idx, colunas_disp]

        # Explicação detalhada para cada recomendado
        explicacoes = []
        for rec_idx in indices_recomendados:
            filme_rec = df.loc[rec_idx]
            # Gêneros
            generos_rec = normaliza_generos(filme_rec['Genre']) if 'Genre' in filme_rec else set()
            generos_comum = generos_base & generos_rec
            # Tags
            if 'Tags' in filme_rec and pd.notna(filme_rec['Tags']) and str(filme_rec['Tags']).strip() != '':
                tags_rec = set([t.strip().lower() for t in str(filme_rec['Tags']).split(',') if t.strip()])
            else:
                tags_rec = set()
            tags_comum = tags_base & tags_rec
            # Diretor
            diretor_rec = filme_rec['Director_clean'] if 'Director_clean' in filme_rec else ''
            diretor_igual = (diretor_base == diretor_rec and diretor_base != '')
            # Stars
            stars_rec = set()
            if 'Stars' in filme_rec:
                stars_rec = split_stars(filme_rec['Stars'])
            stars_comum = stars_base & stars_rec
            diff_rating = abs(filme_rec['IMDb Rating'] - df.loc[idx, 'IMDb Rating']) if 'IMDb Rating' in df.columns else None
            ano_base = int(df.loc[idx, 'Year_Extracted']) if 'Year_Extracted' in df.columns and not pd.isna(df.loc[idx, 'Year_Extracted']) else 0
            ano_rec = int(filme_rec['Year_Extracted']) if 'Year_Extracted' in filme_rec and not pd.isna(filme_rec['Year_Extracted']) else 0
            diff_ano = abs(ano_rec - ano_base) if ano_base > 0 and ano_rec > 0 else None

            pontos_comum = []
            if generos_comum:
                pontos_comum.append(f"Gêneros em comum: {', '.join(generos_comum)}")
            if tags_comum:
                pontos_comum.append(f"Tags em comum: {', '.join(tags_comum)}")
            if diretor_igual:
                pontos_comum.append(f"Mesmo diretor: {filme_rec['Director']}")
            if stars_comum:
                pontos_comum.append(f"Atores em comum: {', '.join(stars_comum)}")
            if not pontos_comum:
                pontos_comum.append("Nenhum ponto forte em comum, mas alta similaridade por outros atributos.")

            explicacao = " | ".join(pontos_comum)
            if diff_rating is not None:
                explicacao += f" | Diferença de rating: {diff_rating:.1f}"
            # Só mostra ano se ambos forem válidos
            if ano_base > 0 and ano_rec > 0:
                if ano_base == ano_rec:
                    explicacao += " | Mesmo ano de lançamento"
                else:
                    explicacao += f" | Diferença de ano: {diff_ano}"
            explicacoes.append(explicacao)

        return filme_original, recomendados, sim_scores, indices_recomendados, explicacoes
    except Exception as e:
        return None, f"Erro na recomendação: {e}", None, None, None

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
                resultado = recomendar_filmes(filme_selecionado, top_k)
                if resultado[0] is not None and resultado[1] is not None:
                    filme_original, recomendacoes, sim_scores, indices, explicacoes = resultado
                    st.success(f"🎯 Filme base: **{filme_original['Title']}** ({filme_original['Year_Extracted']})")
                    # Campos extras
                    info_base = []
                    if 'IMDb Rating' in filme_original:
                        info_base.append(f"⭐ Rating: {filme_original['IMDb Rating']}")
                    if 'Genre' in filme_original:
                        info_base.append(f"🏷️ Gênero: {filme_original['Genre']}")
                    if 'Director' in filme_original:
                        info_base.append(f"🎬 Diretor: {filme_original['Director']}")
                    if 'Stars' in filme_original:
                        info_base.append(f"👥 Elenco: {filme_original['Stars']}")
                    st.write(' | '.join(info_base))

                    st.info("""
**Como funciona a recomendação:**
O sistema utiliza os seguintes campos para calcular a similaridade entre filmes:
- ⭐ IMDb Rating (nota)
- 🏆 Meta Score
- 👥 Número de votos
- 📅 Ano de lançamento
- 🏷️ Gênero (normalizado)
- 🏷️ Tags (se houver)
- 🎬 Diretor (se igual, aumenta similaridade)
- 👥 Elenco (atores em comum aumentam similaridade)
Todos esses campos são normalizados e combinados para encontrar os filmes mais parecidos com o selecionado.
""")

                    # Mostrar a fórmula usada
                    st.subheader("📐 Fórmula da Similaridade do Cosseno")
                    st.latex(r"""
                    	ext{similaridade}(A,B) = 
                    \frac{\sum_{i=1}^n A_i \cdot B_i}
                    {\sqrt{\sum_{i=1}^n A_i^2} \cdot \sqrt{\sum_{i=1}^n B_i^2}}
                    """)
                    st.caption("A representa o vetor de características do filme base e B representa outro filme da base.")

                    st.subheader("🎬 Filmes Recomendados:")
                    for i, (rec_idx, (_, filme), explicacao) in enumerate(zip(indices, recomendacoes.iterrows(), explicacoes), 1):
                        score = sim_scores[i-1]
                        col1, col2, col3 = st.columns([1, 3, 2])
                        with col1:
                            poster_url = filme['Poster URL'] if 'Poster URL' in filme and pd.notna(filme['Poster URL']) and str(filme['Poster URL']).startswith('http') else "https://via.placeholder.com/100x150?text=Filme"
                            st.image(poster_url, width=100)
                        with col2:
                            title = f"**{i}. {filme['Title']}**"
                            ano = filme['Year_Extracted'] if 'Year_Extracted' in filme and pd.notna(filme['Year_Extracted']) and int(filme['Year_Extracted']) > 0 else ''
                            if ano:
                                title += f" ({ano})"
                            st.markdown(title)
                            if 'Genre' in filme and pd.notna(filme['Genre']):
                                st.caption(f"Gênero: {filme['Genre']}")
                            if 'Director' in filme and pd.notna(filme['Director']):
                                st.caption(f"Diretor: {filme['Director']}")
                            if 'Stars' in filme and pd.notna(filme['Stars']):
                                st.caption(f"Elenco: {filme['Stars']}")
                        with col3:
                            st.metric("Rating", filme['IMDb Rating'])
                            st.metric("Similaridade", f"{score:.3f}")
                        st.info(explicacao)
                        st.divider()
                else:
                    st.error(resultado[1])

# ==============================
# 5. Sidebar
# ==============================
if not df.empty:
    st.sidebar.header("ℹ️ Sobre o Sistema")
    st.sidebar.write("""
    Sistema baseado em **Raciocínio Baseado em Casos (RBC)**.
    A recomendação considera múltiplos fatores para garantir resultados realmente similares:
    - ⭐ Avaliação IMDb
    - 🏆 Meta Score (crítica)
    - 👥 Número de votos
    - 📅 Ano de lançamento
    - 🎭 Gênero (normalizado)
    - 🏷️ Tags (temas/palavras-chave)
    - 🎬 Diretor (se igual, aumenta similaridade)
    - 👥 Elenco (atores em comum aumentam similaridade)
    Todos esses campos são combinados para encontrar os filmes mais parecidos com o selecionado.
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

    # Explorar por gênero
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
