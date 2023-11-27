import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever

import os

os.environ["GOOGLE_API_KEY"] = "XXXXXXXXXXXXXXX" # Get it at https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
os.environ["GOOGLE_CSE_ID"] = "XXXXXXXXXXXXXXX" # Get it at https://programmablesearchengine.google.com/

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_API_KEY="XXXXXXXXXXXXXXX"

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

def settings():

    # Vectorstore
    import faiss
    from langchain.vectorstores import FAISS 
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.docstore import InMemoryDocstore  
    embeddings_model = HuggingFaceEmbeddings()  
    embedding_size = 768  # By default, the pretrained models output embeddings with size 768 (base-models) or with size 1024 (large-models). 
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


    # LLM
    from langchain.chat_models import AzureChatOpenAI
    llm = AzureChatOpenAI(deployment_name="XXXXXXXXXXXXXXX-model",
                                openai_api_base="https://XXXXXXXXXXXXXXX.openai.azure.com/",
                                openai_api_version="2023-05-15",
                                openai_api_key=OPENAI_API_KEY,
                                openai_api_type="azure",
                                temperature=0,verbose=True)

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper() 
 

    # Initialize 
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=3
    )

    return web_retriever, llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("image.jpg", width=250)
st.title("LLM with Web Search")
st.info("`I am an AI assistant that can answer questions by exploring the Internet.`")

# Make retriever and llm
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input 
question = st.text_input("Ask anything:")

if question:

    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
    answer.info('`Answer:`\n\n' + result['answer'])
    st.info('`Sources:`\n\n' + result['sources'])
