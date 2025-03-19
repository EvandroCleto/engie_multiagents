# Projeto de- Reasoning, RAG e Multi-Agente Setup (ReAct Agents)

# Imports
import os
import streamlit as st
from phi.agent import Agent, AgentKnowledge
from phi.model.openai import OpenAIChat
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.embedder.openai import OpenAIEmbedder
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.firecrawl import FirecrawlTools
from phi.cli.console import console
from dotenv import load_dotenv
#Importa imagens
from PIL import Image

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
firecrawl_key = os.getenv("FIRECRAWL_API_KEY")

# Configuração da página do Streamlit
st.set_page_config(page_title = "Engie Energia", page_icon = ":100:", layout = "centered")

col1, col2 = st.columns([3.5, 4.5])  #[2.5, 1.5]

# Configuração de colunas para exibir os dados do projeto
with col1:
    image1 = Image.open('imagens/logo.png')
    st.image(image1,width=330)
  
with col2:
  st.title(":blue[Multi-Agent com Reasoning e RAG]")

st.text("")
 
st.markdown('<div style="text-align: center;font-size: 40px"><b>Autor: Evandro Cleto</b></div>', unsafe_allow_html=True)


st.text("")
st.text("")

col4, col5,col6 = st.columns([1.5,3,1.5])  #[2.5, 1.5]

with col4:
    st.title("")
with col5:
    image2 = Image.open('imagens/engie_proj.png')
    st.image(image2,width=400)
with col6:
    st.title("")


# Criando banco vetorial
db_uri = "rag/lancedb"

# Carregar o banco vetorial existente sem reprocessar o PDF
banco_vetorial = PDFUrlKnowledgeBase(
    #urls = ["https://www2.aneel.gov.br/cedoc/ren20241087.pdf"],  # Lista vazia para não baixar PDFs novamente
    urls = [],  # Lista vazia para não baixar PDFs novamente
    vector_db = LanceDb(table_name = "engievectordb",
                        uri = db_uri,
                        search_type = SearchType.vector,
                        embedder = OpenAIEmbedder(model = "text-embedding-3-small")  # Modelo usado na indexação
    )
)

# Load the knowledge base: Comment after first run
#banco_vetorial.load(upsert=True)

# Criando agentes
agente_pesquisa = Agent(
    name = "Agente de Pesquisa",
    role = "Pesquisar na Web",
    model = OpenAIChat(id = "gpt-4o"),
    tools = [DuckDuckGo(fixed_max_results=2)],
    instructions = ["""Você é um especialista juridico no setor de energia do Brasil com 15 anos de experiência 
    em pesquisa do mercado de energia. Seu objetivo é encontrar informações sobre as leis e decretos como:
    2. Lei nº 9.427, de 26 de dezembro de 1996;
    3. Lei nº 10.848, de 15 de março de 2004;
    4. Decreto nº 5.177, de 30 de julho de 2004;
    5. Decreto nº 6.353, de 16 de janeiro de 2008;
    6. Decreto nº 10.707, de 28 de maio de 2021;
    7. Decreto nº 11.835, de 20 de dezembro de 2023.
    Gere a resposta no sempre no idioma Português do Brasil"""],
    markdown = True,
    show_tool_calls = False
)

# Criação do Agente de Web Scraping para busca de informações no site da Engie
agente_webscrping = Agent(
    name = "Agente Web Scraping",
    role = "Fazer web scriping da pagina mercado-livre-de-energia da engie",
    model = OpenAIChat(id = "gpt-4o"),
    tools = [FirecrawlTools(
            api_key = firecrawl_key, 
            scrape=True, 
            crawl=True)
           ],
    instructions = [""" Voce é um especialista que trabalha na Engie Energia com o produto Mercado Livre de Energia e que forncerá todas as resposta sobre este produto
                    que está na página: https://www.engie.com.br/produtos-engie/mercado-livre-de-energia/ .
                    Voce deve demonstrar as vantagens deste produto e considerar as "Perguntas Frequentes" existentes nesta página.
                    Gere a resposta no idioma Português do Brasil"""],
    markdown = True,
    show_tool_calls = True
)
  
  # Print do resumo do webscraping
#agente_webscrping.print_response("Summarize this https://www.engie.com.br/produtos-engie/mercado-livre-de-energia/", stream = True)
  

multi_agente_com_raciocinio_rag = Agent(
    team = [agente_pesquisa, agente_webscrping],
    knowledge = banco_vetorial,
    add_context=True,
    show_tool_calls = True,
    reasoning = True,
    markdown = True,
    structured_outputs = True
)

# Prompt do usuário
tarefa = st.text_area("📝 Insira Sua Pergunta Para os Agentes de IA:", 
                      "Analise a RESOLUÇÃO NORMATIVA ANEEL Nº 1.087 e verifique se  o produto 'mercado livre de energia' da engie energia está em conformidade com esta resolução. Gere sempre as resposta no idioma portugues do Brasil.")

if st.button("Executar Análise"):
    with st.spinner("⏳ Os Agentes de IA Estão Processando Sua Consulta. Aguarde..."):
        try:
            response = multi_agente_com_raciocinio_rag.run(tarefa)
            
            # Debug: Verificar estrutura da resposta
            #st.write("Debug - Tipo da resposta:", type(response))
            #st.write("Debug - Conteúdo bruto da resposta:", response)

            # Extraindo somente o conteúdo relevante
            if hasattr(response, "get_content_as_string"):
                resposta_formatada = response.get_content_as_string() 
            elif hasattr(response, "content") and response.content is not None:
                resposta_formatada = response.content 
            else:
                resposta_formatada = "Erro: Resposta não contém conteúdo válido."

            st.markdown("### 📌 Resposta dos Agentes:")
            st.write(resposta_formatada)

        except Exception as e:
            st.error(f"Erro ao processar a tarefa: {e}")




