# engie_multiagents
# Projeto de Reasoning, RAG, Multi-Agente Setup (ReAct Agents) com WebScraping

Resumo do Projeto: Projeto de multiagentes com raciocínio, RAG usando LanceDB para armzenamento de arquivo pdf e web screping para baixar pagina de um site e com front-end usando Streamilt.
O projeto é composto de:
1- Pacote PDFUrlKnowledgeBase para baixar pdf de legislação e gravá-lo no LanceDB;
2- Agente construído com o pacote phidata para pesquisa de legislações na internet usando DuckDuckGo;
3- Agente construído com o pacote phidata para web screping de página na internet que descreve um serviço de uma empresa;
4- Multi Agente orquestrador com racíocinio e RAG no Lance DB construído com o pacote phidata;
5- Front-end usando Streamilt para realizar pesquisa utilizando o Multi Agente;
6- O LLM utilido foi o gpt-4o.
