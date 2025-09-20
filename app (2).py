import streamlit as st
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from groq import Groq
import PyPDF2
import docx
from io import BytesIO
import tempfile

# Set up the page
st.set_page_config(
    page_title="RAG App with Groq",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# Initialize Groq client
@st.cache_resource
def init_groq_client():
    # Try to get API key from environment variable first, then from Streamlit secrets
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("❌ Groq API key not found! Please set it in environment variables or Streamlit secrets.")
        st.stop()
    
    return Groq(api_key=api_key)

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_embeddings_and_index(documents):
    """Create embeddings and FAISS index for documents"""
    if not documents:
        return None, None
    
    model = load_embedding_model()
    
    # Create embeddings
    with st.spinner("Creating embeddings..."):
        embeddings = model.encode(documents, show_progress_bar=False)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    return embeddings, index

def retrieve_relevant_chunks(query, index, documents, k=3):
    """Retrieve most relevant document chunks"""
    if index is None or not documents:
        return []
    
    model = load_embedding_model()
    
    # Encode query
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding.astype('float32'), k)
    
    # Return relevant chunks with scores
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(documents):
            results.append({
                'text': documents[idx],
                'score': float(score),
                'rank': i + 1
            })
    
    return results

def generate_response(query, context_chunks):
    """Generate response using Groq"""
    client = init_groq_client()
    
    # Prepare context
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    
    # Create prompt
    prompt = f"""Based on the following context, please answer the question. If the answer is not available in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. Be accurate and concise."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
def main():
    st.title("🤖 RAG App with Groq & Sentence Transformers")
    st.markdown("Upload documents and ask questions based on their content!")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Support for TXT, PDF, and DOCX files"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    
                    for file in uploaded_files:
                        st.write(f"Processing: {file.name}")
                        
                        if file.type == "text/plain":
                            text = str(file.read(), "utf-8")
                        elif file.type == "application/pdf":
                            text = extract_text_from_pdf(file)
                        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text = extract_text_from_docx(file)
                        else:
                            st.error(f"Unsupported file type: {file.type}")
                            continue
                        
                        if text:
                            chunks = chunk_text(text)
                            all_chunks.extend(chunks)
                            st.success(f"✅ {file.name}: {len(chunks)} chunks created")
                    
                    if all_chunks:
                        # Store in session state
                        st.session_state.documents = all_chunks
                        
                        # Create embeddings and index
                        embeddings, index = create_embeddings_and_index(all_chunks)
                        st.session_state.embeddings = embeddings
                        st.session_state.index = index
                        
                        st.success(f"🎉 Successfully processed {len(all_chunks)} document chunks!")
                    else:
                        st.error("No text could be extracted from the uploaded files.")
        
        # Display document stats
        if st.session_state.documents:
            st.success(f"📊 Ready! {len(st.session_state.documents)} chunks loaded")
            
            if st.button("Clear Documents"):
                st.session_state.documents = []
                st.session_state.embeddings = None
                st.session_state.index = None
                st.rerun()
    
    # Main area for Q&A
    st.header("💬 Ask Questions")
    
    if not st.session_state.documents:
        st.info("👆 Please upload and process some documents first using the sidebar.")
        
        # Demo section
        st.markdown("---")
        st.subheader("🚀 Demo Mode")
        st.markdown("You can also try a quick demo with sample text:")
        
        if st.button("Load Sample Documents"):
            sample_docs = [
                "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
                "Computer vision is a field of AI that trains computers to interpret and understand the visual world.",
                "Reinforcement learning is a type of machine learning where agents learn to make decisions by taking actions in an environment."
            ]
            
            st.session_state.documents = sample_docs
            embeddings, index = create_embeddings_and_index(sample_docs)
            st.session_state.embeddings = embeddings
            st.session_state.index = index
            st.success("Sample documents loaded! Try asking about AI topics.")
            st.rerun()
    else:
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="Ask anything about your uploaded documents...",
            help="Type your question and press Enter"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_button = st.button("🔍 Search & Answer", type="primary")
        
        with col2:
            num_chunks = st.selectbox("Chunks to retrieve:", [3, 5, 7], index=0)
        
        if search_button and query:
            with st.spinner("Searching and generating answer..."):
                # Retrieve relevant chunks
                relevant_chunks = retrieve_relevant_chunks(
                    query, 
                    st.session_state.index, 
                    st.session_state.documents, 
                    k=num_chunks
                )
                
                if relevant_chunks:
                    # Generate response
                    response = generate_response(query, relevant_chunks)
                    
                    # Display results
                    st.markdown("### 🎯 Answer")
                    st.markdown(response)
                    
                    # Show retrieved chunks
                    st.markdown("---")
                    st.markdown("### 📚 Retrieved Context")
                    
                    for i, chunk in enumerate(relevant_chunks, 1):
                        with st.expander(f"Chunk {i} (Similarity: {chunk['score']:.3f})"):
                            st.text(chunk['text'])
                else:
                    st.error("No relevant information found in the documents.")
        
        # Example queries
        if st.session_state.documents:
            st.markdown("---")
            st.markdown("### 💡 Example Queries")
            
            examples = [
                "What is the main topic discussed?",
                "Can you summarize the key points?",
                "What are the important details mentioned?"
            ]
            
            cols = st.columns(len(examples))
            for i, example in enumerate(examples):
                with cols[i]:
                    if st.button(example, key=f"example_{i}"):
                        st.session_state.temp_query = example

if __name__ == "__main__":
    main()
  
   