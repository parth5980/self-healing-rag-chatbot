import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()

# Initialize LLM and Embeddings models
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7
)

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

# making Pinecone searver if user not creted it automatically create index in pincone searver
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Load and split documents
def load_and_split(source_type: str, source: str) -> list:
    """Load documents from any source and split into chunks"""
    
    text_splitter = RecursiveCharacterTextSplitter(#chat spliter use for making split letter by letter and line by line and paragraph by paragraph
        chunk_size=1000,
        chunk_overlap=200
    )
    
    if source_type == "pdf":      # if  is pdf
        loader = PyPDFLoader(source)
        documents = loader.load()
    elif source_type == "url":   #if  is url
        loader = WebBaseLoader(source)
        documents = loader.load()
    elif source_type == "youtube":   #if  is you tube link
        loader = YoutubeLoader.from_youtube_url(source)
        documents = loader.load()
    elif source_type == "text":    #if  is texts,pargraph or anything like that
        documents = [Document(page_content=source)]
    else:
        raise ValueError(f"Invalid source_type: {source_type}")
    
    chunks = text_splitter.split_documents(documents)
    return chunks


# Store chunks in Pinecone
def store_chunks(chunks: list) -> None:
    """Delete old vectors and store new chunks in Pinecone"""
    
    index = pc.Index(index_name)
    index.delete(delete_all=True)
    
    global vectorstore, retriever
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Format docs helper
def format_docs(docs: list) -> str:   #Goes through each doc and joins them and returm 1 string
    """Join chunks into one string"""
    return "\n\n".join(doc.page_content for doc in docs)


# Convert history to string
def history_str(chat_history: list) -> str:
    """Convert chat history to plain string"""
    history = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            history += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history += f"AI: {msg.content}\n"
    return history

# Global variables
vectorstore = None
retriever = None
chain = None

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are DocuMind - a helpful AI assistant.
Answer the question based on the context provided.
If answer is not in context, say "I don't have enough information about this in the document."

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:
""")

# Build LCEL chain
def build_chain(retriever):
    """Builds LCEL pipeline"""
    lcel_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return lcel_chain

# Update store_chunks to also build chain
def store_chunks(chunks: list) -> None:
    """Delete old vectors and store new chunks in Pinecone"""
    index = pc.Index(index_name)
    index.delete(delete_all=True)
    global vectorstore, retriever, chain
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = build_chain(retriever)

# Process source - full pipeline
def process_source(source_type: str, source: str) -> dict:
    """
    Full pipeline - load, split and store
    Called when user uploads document
    
    Args:
        source_type: "pdf", "url", "youtube", "text"
        source: file path, URL, or plain text
    Returns:
        dict with message and chunk count
    """
    chunks = load_and_split(source_type, source)
    store_chunks(chunks)
    clear_history()
    return {
        "message": "Document loaded successfully!",
        "chunks": len(chunks)
    }

# Chat history
chat_history = [] #for making the llm to rember past coverstation of ours

# Simple chat
def chat(question: str) -> str:
    """Chat with document"""
    response = chain.invoke({
        "question": question,
        "chat_history": history_str(chat_history)
    })
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response

# Grade chunks
def grade_chunks(question: str, chunks: list) -> float: #for grading kwargs to rate it out of 5 and return its lable
    """Grade retrieved chunks out of 5.0"""
    chunks_text = format_docs(chunks)
    grade_prompt = f"""
    Question: {question}
    Retrieved context: {chunks_text}
    Rate relevance 1.0-5.0 (0.5 steps only). 
    Decimal number only:
    """
    result = llm.invoke(grade_prompt)
    score_text = result.content.strip()
    try:
        score = float(score_text[:3])
        score = max(1.0, min(5.0, score))
    except:
        score = 3.0
    return score

# Get label
def get_label(score: float) -> str:  #for btter scoring and gives lables to graded chunks
    """Convert score to label"""
    if score >= 4.5:
        return "🌟 Excellent"
    elif score >= 4.0:
        return "✅ Good"
    elif score >= 3.0:
        return "🔄 Average"
    elif score >= 2.0:
        return "⚠️ Poor"
    else:
        return "❌ Very Poor"

# Self healing chat
def self_healing_chat(question: str) -> tuple:   # if grade is not good like its grade below 3/5 then it retry and find better result
    """Chat with automatic self-healing"""
    current_question = question
    for attempt in range(1, 4):
        chunks = retriever.invoke(current_question)
        score = grade_chunks(current_question, chunks)
        label = get_label(score)
        if score >= 3.0:
            break
        current_question = question + " explain in detail with examples"
    response = chain.invoke({
        "question": question,
        "chat_history": history_str(chat_history)
    })
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response, score, label, attempt

# Clear history
def clear_history() -> None:   #after remvoing histry from chat
    """Clear chat history"""
    chat_history.clear()