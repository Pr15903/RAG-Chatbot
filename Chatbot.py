import os
import warnings
import logging
import time
from typing import Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, MessagesPlaceholder
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader, PDFPlumberLoader,
    UnstructuredPowerPointLoader
)
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware

# Suppress warnings and configure logging
warnings.filterwarnings("ignore")
logging.getLogger("chromadb.config").setLevel(logging.ERROR)

# FastAPI App Initialization
app = FastAPI(title="AI Assistant API")

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())
print("Torch Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Use specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# FastAPI Request Model
class QueryRequest(BaseModel):
    query: str


PROPMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the user's query **EXACTLY as per the given format**.

---

### **Answering Rules:**
1. **Analyze the context first** before answering. Ensure your response is based **ONLY on the provided context**—do **NOT** use external knowledge.
2. **Strictly follow the required response format**:
   - If the user asks for a **command**, return only the exact command.
   - If the user asks for **steps**, return them in **flowchart format** (see below).
   - If the user asks for a **list**, return it as **bullet points**.
   - If unsure, respond with: **"I don't know"**.
3. **DO NOT** repeat the user's query in the answer.
4. **DO NOT** add extra details, explanations, or assumptions unless explicitly requested.
5. **Max 3 sentences unless explicitly asked for more**.
6. **If the query is a greeting (e.g., "Hello", "Hi"), respond with a greeting**.
7. **If asked for an example, provide only what is requested.**
8. **If the context lacks the required information, respond with: `"I don't know"`**.

---

### **Example Query & Response Formatting:**

#### **1. Command Requests (Example)**
Query: **"Print command for Excel report?"**  
Response:
        <CMD=print|Value=now|DateFormat=dd/MM/yyyy hh:mm:ss.fff>

Query: **"Query command in Excel report?"**  
Response:
        <CMD=query|Value=SELECT * FROM [XStudio_Historian].[dbo].[XHS_Datatable_Mst_Tbl] WHERE DatabaseName IN (SELECT Name FROM [XStudio_Historian].[dbo].[XHS_Database_Mst_Tbl] WHERE StartDate BETWEEN '{{runtime.now(dd-MMM-yyyy 12:00:00)}}' AND '{{runtime.now(dd-MMM-yyyy 15:00:00)}}')|IsHeader=1|IsAutoFit=0|mode=Copy>
        
---

#### **2. Flowchart Requests (Only if asked for steps)**
- **Use this format**:
[START] --> [Step 1] --> [Step 2] --> [END] │
└─> [Alternative Step]

- Example:
Query: **"Steps to restart a frozen app?"**  
Response:
        [START] --> Open Task Manager --> Select app --> End Task and Relaunch app. --> [END]

---

#### **3. List Requests (Features, Properties, etc.)**
- **Use bullet points with NO extra details**.
- Example:
Query: **"List features of the Query Command?"**  
Response: 
        > Supports SQL queries.
        > Allows enabling/disabling headers.
        > Provides auto-fit column adjustment.
        > Supports Copy and Insert modes.

---

#### **4. Greeting Requests**
Query: **"Hi"**  
Response:
        Hello! How can I help you?
---

#### **5. Unavailable Information**
Query: **"How do I configure a database connection?"**  
*(If the context does not provide the answer)*  
Response:
        I don't know.

---

### **Final Notes**
- **Use ONLY the information from the given context.**
- **Do NOT assume anything beyond the provided details.**
- **If unsure, reply with "I don't know".**

---        

"""

# Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="user_query")

# LLM Model Configuration
LANGUAGE_MODEL = OllamaLLM(model="phi3:latest", temperature=0.5)

# Document Embedding Model Class
class EmbeddingModel:
    def __init__(self):
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
            self.vector_store = None
            all_chunked_documents = []
            self.indexed_chunk_ids = set()

            for file_name in os.listdir("D:/Developers/Rohan Vesuwala/Projects/Demo/Xstudio Help Assistant/Xstudio Help Assistant/wwwroot/uploads"):
                file_path = f"D:/Developers/Rohan Vesuwala/Projects/Demo/Xstudio Help Assistant/Xstudio Help Assistant/wwwroot/uploads/{file_name}"
                if os.path.exists(file_path):
                    document = self.load_file(file_path)
                    if document:
                        chunked_documents = self.chunk_documents(document)
                        all_chunked_documents.extend(chunked_documents)
                        self.index_documents(all_chunked_documents, file_name)
        except Exception as e:
            print("Error in loading model", e)

    def load_file(self, file_path: str):
        if file_path.endswith(".pdf"):
            return PDFPlumberLoader(file_path).load()
        elif file_path.endswith(".pptx"):
            return UnstructuredPowerPointLoader(file_path).load()
        else:
            return UnstructuredWordDocumentLoader(file_path).load()

    def chunk_documents(self, document):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, separators=["\n\n", "\n", " "])
        return text_splitter.split_documents(document)

    def index_documents(self, all_chunked_documents, file_name):
        try:
            new_docs = []
            for i, doc in enumerate(all_chunked_documents):
                chunk_id = f"{file_name}_chunk_{i}"
                if chunk_id in self.indexed_chunk_ids:
                    continue
                new_docs.append(Document(page_content=doc.page_content, metadata={"source": file_name, "chunk_id": chunk_id}))
            
            if new_docs:
                if os.path.exists("./faiss_index"):
                    self.vector_store = FAISS.load_local("./faiss_index", self.embedding_model, allow_dangerous_deserialization=True)
                    self.vector_store.add_documents(new_docs)
                else:
                    self.vector_store = FAISS.from_documents(new_docs, self.embedding_model)
                self.vector_store.save_local('./faiss_index')
                for doc in new_docs:
                    self.indexed_chunk_ids.add(doc.metadata["chunk_id"])
        except Exception as e:
            print("Error in indexing document:", e)

    def find_related_doc(self, query):
        if not self.vector_store:
            return []
        results = self.vector_store.max_marginal_relevance_search(query, fetch_k=10, k=3, lambda_mult=0.5)
        return [{"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")} for doc in results]

file_processor = EmbeddingModel()
@app.get("/Index")
def Indexed():
    return 'Hello',200;

@app.post("/query")
def query_document(request: QueryRequest) -> Dict[str, Any]:
    try:
        prompt = request.query
        if prompt:
            related_docs = file_processor.find_related_doc(prompt)

            if not related_docs:
                return {"response": "No related document found", "sources": [], "response_time": 0}

            logging.info("Get docuemnt")
            context_text = "\n\n".join(doc["content"] for doc in related_docs)
            sources = list(set(doc["source"] for doc in related_docs))

            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(PROPMPT_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                SystemMessagePromptTemplate.from_template("{document}"),
                HumanMessagePromptTemplate.from_template("{user_query}")
            ]);

            logging.info("Createting chain")
            response_chain = LLMChain(llm=LANGUAGE_MODEL, prompt=prompt_template)
            chat_history = memory.load_memory_variables({}).get("chat_history", [])
            start_time = time.time()
            logging.info("Given to model")
            response = response_chain.invoke({"chat_history": chat_history, "document": context_text, "user_query": prompt});
            end_time = time.time()
            logging.info("Save to chat histroy")
            memory.save_context({"user_query": prompt}, {"Answer": response['text']})
            return {"response": response['text'], "sources": sources, "response_time": end_time - start_time}
        else:
            return "No prompt given", 404;

    except Exception as e:
        import traceback
        return traceback.print_exc(), 404;