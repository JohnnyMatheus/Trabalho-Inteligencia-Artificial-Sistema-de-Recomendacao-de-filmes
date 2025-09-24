<h1 align="center">Inteligência Artificial - Sistema de Recomendação de Filmes (RBC)</h1>

## <p align="center">👨🏽‍🎓Nome completo: Johnny Matheus Nogueira de Medeiro, Nathaniel Nicolas Rissi Soares, Nelson Ramos Rodrigues Junior</p>
## <p align="center">🏫Turma: Ciências da Computação UNOESC - São Miguel do Oeste</p>
<hr />

<p align="justify">
  O projeto desenvolvido consiste em um sistema de recomendação de filmes baseado em Raciocínio Baseado em Casos (RBC), criado utilizando modernas ferramentas de desenvolvimento e conceitos de inteligência artificial, aplicando todo o conhecimento adquirido no componente curricular Inteligência Artificial do curso de Ciência da Computação da Universidade do Oeste de Santa Catarina (UNOESC), ministrado pelo professor Vinicius Almeida dos Santos. O objetivo principal do projeto é aplicar na prática os conhecimentos adquiridos em sala de aula, fornecendo recomendações personalizadas de filmes com base em avaliações, votos e ano de lançamento.
</p>

<p>O sistema foi implementado utilizando Python, Pandas, Numpy, Scikit-learn, Streamlit e ferramentas de visualização interativa, proporcionando uma experiência prática e didática na aplicação de técnicas de inteligência artificial em dados reais, no contexto do curso de Ciência da Computação.</p>

## Objetivo do Projeto
--O projeto é um Sistema de Recomendação de Filmes baseado em Raciocínio Baseado em Casos (RBC).
-- RBC funciona como o raciocínio humano: você compara um novo caso (filme que o usuário gosta) com casos anteriores (outros filmes da base) e recomenda os mais similares.
-- O objetivo é ajudar o usuário a descobrir filmes parecidos com os que ele já gosta, usando informações numéricas e características dos filmes.

## Base de Dados

O sistema usa um arquivo CSV com informações dos filmes, como:
Title → Nome do filme
IMDb Rating → Nota no IMDb
Meta Score → Nota da crítica
Votes → Número de votos do público
Year → Ano de lançamento
Genre → Gênero(s) do filme

## Pré-processamento
Antes de recomendar, os dados passam por algumas etapas:
Limpeza de votos → transforma valores com "K" em números inteiros.
Extração do ano → pega o ano do título do filme.
Seleção de features numéricas → IMDb Rating, Meta Score, votos e ano.
Normalização → usa MinMaxScaler para colocar todas as características na mesma escala (0 a 1), o que é importante para calcular similaridade corretamente.

## Algoritmo de Recomendação
O coração do projeto é a função de recomendação:
O usuário seleciona um filme na interface.
O sistema encontra o filme na base de dados (busca parcial e case insensitive).
Calcula a similaridade de cosseno entre o filme escolhido e todos os outros da base.
Similaridade de cosseno: mede o quão parecidos dois filmes são considerando seus valores numéricos (rating, votos, ano…).
Retorna os top K filmes mais similares, ignorando o próprio filme selecionado.
Exibe para o usuário:
Filme base (o que ele selecionou)
Filmes recomendados, com nota e gênero
Resumindo: ele pega “um filme que você gosta” e acha os mais parecidos na base usando matemática de vetores.

## Interface do Streamlit
O projeto usa Streamlit para criar uma interface interativa:
Selectbox → permite escolher um filme.
Slider → define quantas recomendações mostrar.
Checkboxes → mostrar dados brutos ou debug.
Sidebar → mostra estatísticas da base e permite explorar filmes por gênero.
Tudo isso é feito de forma visual e dinâmica, sem precisar mexer no código.

## Tratamento de Erros
O sistema é robusto, por isso:
Se o CSV não existir → mostra mensagem de erro.
Se uma coluna faltar → usa apenas as colunas disponíveis.
Se o filme não for encontrado → avisa o usuário.
Isso evita que o sistema quebre durante a execução, garantindo uma apresentação segura.

## 🧠 Desenvolvedores

| [<img src="https://avatars.githubusercontent.com/u/128015032?v=4" width=115><br>👑Game Master👑<br><sub>🐦‍🔥Johnny Matheus Nogueira de Medeiro🐦‍🔥</sub>](https://github.com/JohnnyMatheus) | [<img src="https://avatars.githubusercontent.com/u/166051346?v=4" width=115><br><sub>Nelson Ramos Rodrigues Junior</sub>](#) | [<img src="https://avatars.githubusercontent.com/u/165223471?v=4" width=115><br><sub>Nathaniel Nicolas Rissi Soares</sub>](#) |
| :---: | :---: | :---: |


## 🔷 Professor

| [<img src="https://avatars.githubusercontent.com/u/7074409?v=4" width=115><br><sub>Roberson Alves</sub>](https://github.com/ViniciusAS) |
| :---: |

