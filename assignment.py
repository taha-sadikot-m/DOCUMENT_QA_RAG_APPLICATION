import os
import torch
import warnings


torch.classes.__path__ = []
warnings.filterwarnings("ignore", category=UserWarning)


if torch.cuda.is_available():
    torch.set_default_device('cpu')

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
import numexpr
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document


DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 4000  
CHUNK_OVERLAP = 200
VECTOR_STORE_PATH = "vector_store"
MODEL_NAME = "llama3-70b-8192"  
ALLOWED_FILE_TYPES = [".txt", ".pdf", ".docx"]  


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu",
        "trust_remote_code": True,
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 8 
    }
)

def initialize_vector_store():
    """Handle vector store initialization with error handling"""
    documents = []
    
 
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
 
    files_exist = False
    for filename in os.listdir(DOCUMENTS_DIR):
        if os.path.splitext(filename.lower())[1] in ALLOWED_FILE_TYPES:
            files_exist = True
            break
            

    if not files_exist:
        st.info("No documents found in the documents folder. Starting with an empty knowledge base.")
 
        documents = [Document(page_content="This is a placeholder document.", metadata={"source": "placeholder"})]
        return FAISS.from_documents(documents, embeddings)
    
 
    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        file_ext = os.path.splitext(filename.lower())[1]
        
        if file_ext not in ALLOWED_FILE_TYPES:
            continue
            
        try:
            if file_ext == ".txt":
                loader = TextLoader(filepath)
            elif file_ext == ".pdf":
                loader = PyPDFLoader(filepath)
            elif file_ext == ".docx":
                loader = UnstructuredWordDocumentLoader(filepath)
            else:
                continue
                
            loaded = loader.load()
            documents.extend(loaded)
            st.toast(f"Loaded {len(loaded)} documents from {filename}")
        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")
            continue


    if not documents:
        st.warning("No documents found. Using empty knowledge base.")
        documents = [Document(page_content="Empty KB", metadata={"source": "generated"})]


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)


    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store


try:
 
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        try:
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.success("Loaded existing vector store.")
        except Exception as load_error:
            st.warning(f"Failed to load existing vector store: {str(load_error)}. Creating new one.")
            vector_store = initialize_vector_store()
            st.success("Created new vector store.")
    else:
   
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
   
        vector_store = initialize_vector_store()
        st.success("Created new vector store.")
except Exception as e:
    st.error(f"Vector store initialization failed: {str(e)}")
    st.stop()


try:

    try:
    
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        #st.success("Using API key from Streamlit secrets")
    except Exception:
        try:
            # Then try environment variable
            import os
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            if GROQ_API_KEY:
                #st.success("Using API key from environment variable")
                pass;
            else:
                # Finally fallback to hardcoded key for development
                GROQ_API_KEY = "YOUR-API-KEY"
               # st.warning("Using fallback API key - for production, set in secrets or environment")
        except Exception:
            # Last resort, use hardcoded key
            GROQ_API_KEY = "YOUR-API-KEY"
            #st.warning("Using fallback API key - for production, set in secrets or environment")
    
    # Initialize the LLM with the API key we found
    llm = ChatGroq(
        temperature=0.1,
        model=MODEL_NAME,     
        api_key=GROQ_API_KEY  
    )
    
    #st.success("Successfully initialized Groq client")
except Exception as e:
    st.error(f"Groq client initialization failed: {str(e)}")
    st.stop()


def calculate(expression: str) -> str:
    try:
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Calculation error: {str(e)}"

calculator_tool = Tool(
    name="Calculator",
    func=calculate,
    description="Useful for mathematical calculations"
)

dictionary = {
    "llm": "Large Language Model: A type of AI system trained on vast amounts of text data",
    "rag": "Retrieval-Augmented Generation: Combines information retrieval with text generation"
}

def define_term(term: str) -> str:
    return dictionary.get(term.lower(), "Definition not found in knowledge base.")

dictionary_tool = Tool(
    name="Dictionary",
    func=define_term,
    description="Useful for defining technical terms"
)

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}
"""
)


def process_query(query):
    result = {"tool": "", "context": "", "answer": "", "log": []}
    
    try:
        if "calculate" in query.lower():
            result["tool"] = "Calculator"
            result["log"].append("Detected calculation request")
            expression = query.lower().split("calculate")[-1].strip()
            result["answer"] = calculator_tool.run(expression)
            
        elif "define" in query.lower():
            result["tool"] = "Dictionary"
            result["log"].append("Detected definition request")
            term = query.lower().split("define")[-1].strip()
            result["answer"] = dictionary_tool.run(term)
            
        else:
            result["tool"] = "RAG Pipeline"
            result["log"].append("Initiating RAG process")
            docs = vector_store.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            result["context"] = context
            
            prompt = prompt_template.format(context=context, question=query)
            response = llm.invoke(prompt)
            result["answer"] = response.content
            result["log"].append(f"Retrieved {len(docs)} context chunks")
    
    except Exception as e:
        result["answer"] = f"Error processing query: {str(e)}"
        result["log"].append(f"Error: {str(e)}")
    
    return result


st.title("🧠 Document  Q&A")


with st.sidebar:
    st.header("Document Management")
    

    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["txt", "pdf", "docx"], 
        accept_multiple_files=True
    )
    

    if uploaded_files:
        for uploaded_file in uploaded_files:
           
            save_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
            
          
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Saved: {uploaded_file.name}")
        
      
        if st.button("Rebuild Knowledge Base"):
            with st.spinner("Building knowledge base..."):
                try:
                    vector_store = initialize_vector_store()
                    st.success("Knowledge base updated successfully!")
                except Exception as e:
                    st.error(f"Failed to update knowledge base: {str(e)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_details" not in st.session_state:
    st.session_state.chat_details = []


for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i // 2 < len(st.session_state.chat_details):
            details = st.session_state.chat_details[i // 2]
            with st.expander("View Details"):
                st.write(f"**Tool used:** {details['tool']}")
                if details["context"]:
                    st.write("**Retrieved Context:**")
                    st.write(details["context"])
                st.write("**Processing Log:**")
                st.write("\n".join(details["log"]))


query = st.chat_input("Ask a question...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            result = process_query(query)
            
        message_placeholder.markdown(result["answer"])
        
     
        st.session_state.chat_details.append({
            "tool": result["tool"],
            "context": result["context"],
            "log": result["log"]
        })
    
 
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
