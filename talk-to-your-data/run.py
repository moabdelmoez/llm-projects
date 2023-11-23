import os
import openai
# from openai.embeddings_utils import get_embedding
# from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain.embeddings.cohere import CohereEmbeddings

os.environ["COHERE_API_KEY"] = "XXXXXXXXX"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai.api_type = "azure"
openai.api_base = "https://XXXXXXX.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
OPENAI_API_KEY="XXXXXXX"
openai.api_key = os.getenv("OPENAI_API_KEY")


######
def get_faiss_vectordb(file: str):
    # Extract the filename and file extension from the input 'file' parameter.
    filename, file_extension = os.path.splitext(file)

    # Initiate embeddings using OpenAI.
    # embedding = OpenAIEmbeddings(openai_api_type='azure', model="text-embedding-ada-002", deployment='azure_embeddings', openai_api_key=OPENAI_API_KEY, chunk_size=16)

    # embedding = HuggingFaceEmbeddings()
    embedding = CohereEmbeddings(model = "multilingual-22-12")

    # Create a unique FAISS index path based on the input file's name.
    faiss_index_path = f"faiss_index_{filename}"

    # Determine the loader based on the file extension.
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path=file)
    elif file_extension == ".txt":
        loader = TextLoader(file_path=file)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path=file)
    else:
        # If the document type is not supported, print a message and return None.
        print("This document type is not supported.")
        return None

    # Load the document using the selected loader.
    documents = loader.load()

    # Split the loaded text into smaller chunks for processing.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separators=["\n", "\n\n", "(?<=\. )", "", " "],
    )
    doc_chunked = text_splitter.split_documents(documents=documents)

    # Create a FAISS vector database from the chunked documents and embeddings.
    vectordb = FAISS.from_documents(doc_chunked, embedding)
    
    # Save the FAISS vector database locally using the generated index path.
    vectordb.save_local(faiss_index_path)
    
    # Return the FAISS vector database.
    return vectordb
######

##########
def run_llm(vectordb, query: str) -> str:
    # Create an instance of the ChatOpenAI with specified settings.
    openai_llm = AzureChatOpenAI(deployment_name="MODEL_NAME",
                                openai_api_base="https://XXXXXXX.openai.azure.com/",
                                openai_api_version="2023-05-15",
                                openai_api_key=OPENAI_API_KEY,
                                openai_api_type="azure",
                                temperature=0,verbose=True)
    
    # Create a RetrievalQA instance from a chain type with a specified retriever.
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=openai_llm, chain_type="stuff", retriever=vectordb.as_retriever()
    )
    
    # Run a query using the RetrievalQA instance.
    answer = retrieval_qa.run(query)
    
    # Return the answer obtained from the query.
    return answer
##########


#####################
# Import the necessary libraries.
import streamlit as st

from PIL import Image

image = Image.open('chat.png')

st.image(image)

# Set the title for the Streamlit app.
# st.title("📝 Talk to Your Document")

# Allow the user to upload a file with supported extensions.
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

# Provide a text input field for the user to ask questions about the uploaded article.
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

# If an uploaded file is available, process it.
if uploaded_file:
    # Save the uploaded file locally.
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Create a FAISS vector database from the uploaded file.
    vectordb = get_faiss_vectordb(uploaded_file.name)
    
    # If the vector database is not created (unsupported file type), display an error message.
    if vectordb is None:
        st.error(
            f"The {uploaded_file.type} is not supported. Please load a file in pdf, txt, or md"
        )

# Display a spinner while generating a response.
with st.spinner("Generating response..."):
    # If both an uploaded file and a question are available, run the model to get an answer.
    if uploaded_file and question:
        answer = run_llm(vectordb=vectordb, query=question)
        # Display the answer in a Markdown header format.
        st.write("### Answer")
        st.write(f"{answer}")

################
