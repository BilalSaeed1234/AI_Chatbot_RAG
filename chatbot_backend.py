
import os
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv
import hashlib
import tempfile
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please check your .env file")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

st.title('📚 Bilal PDF Chatbot')
st.caption("Powered by Gemini 1.5 Flash - Upload any PDF and ask questions")

# Setup session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

def get_file_hash(file_content, filename):
    """Generate unique hash for file content and name"""
    hash_data = f"{filename}_{len(file_content)}_{hashlib.md5(file_content).hexdigest()}"
    return hashlib.md5(hash_data.encode()).hexdigest()

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF file and create vector store (in-memory Chroma)"""
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Debug: Show a sample of the loaded content
        with st.expander("🔍 Debug: Loaded PDF Content"):
            sample_content = "\n".join([doc.page_content[:200] for doc in documents[:2]])
            st.write(f"**Sampled Content from {uploaded_file.name}:**")
            st.text(sample_content[:1000] + "..." if len(sample_content) > 1000 else sample_content)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Create in-memory Chroma vector store
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=None)

        # Clean up temporary file
        os.unlink(tmp_path)

        return vectorstore, uploaded_file.name

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None

def get_gemini_response(query, context, pdf_name="the document"):
    """Get response from Gemini 1.5 Flash model using RAG context"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""You are a helpful AI assistant. Use the following context from {pdf_name} to answer the question accurately.

Context:
{context}

Question: {query}

Provide a direct, comprehensive answer based ONLY on the context. 
If the context doesn't contain relevant information, say "I don't have enough information in the document to answer this question."
Do not make assumptions or use external knowledge beyond the provided context.

Answer:"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.7,
                top_k=30,
                max_output_tokens=2048,
            )
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def detect_document_type(vectorstore, pdf_name):
    """Detect document type based on content using Gemini"""
    try:
        # Retrieve a sample of the document content
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        sample_docs = retriever.get_relevant_documents("Summarize the document content")
        context = "\n\n".join([doc.page_content[:500] for doc in sample_docs])

        # Debug: Show the context used for document type detection
        with st.expander("🔍 Debug: Document Type Detection Context"):
            st.write(f"**Sampled Content for {pdf_name}:**")
            st.text(context[:1000] + "..." if len(context) > 1000 else context)

        # Use Gemini to classify document type
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Based on the following content from {pdf_name}, classify the document type as one of: 
        'Curriculum Vitae (CV)', 'NLP Training', 'Report', 'Research Paper', or 'Other'. Provide only the document type as the response.

        Content:
        {context}
        """

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.7,
                top_k=30,
                max_output_tokens=50,
            )
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error detecting document type: {str(e)}")
        return "Unknown"

# Sidebar for PDF upload and management
with st.sidebar:
    st.header("📁 Upload PDF Document")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF document to ask questions about it"
    )

    if uploaded_file is not None:
        # Generate unique hash for the uploaded file
        current_file_hash = get_file_hash(uploaded_file.getvalue(), uploaded_file.name)

        # Check if this is a new file
        is_new_file = (st.session_state.file_hash != current_file_hash)

        if st.button("🚀 Process Document", key="process_btn") or is_new_file:
            with st.spinner("Processing PDF document..."):
                # Clear old vectorstore
                st.session_state.vectorstore = None

                vectorstore, filename = process_uploaded_pdf(uploaded_file)
                if vectorstore:
                    # Reset old state
                    st.session_state.current_pdf = None
                    st.session_state.uploaded_file = None
                    st.session_state.file_hash = None
                    st.session_state.messages = []

                    # Save new state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_pdf = filename
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.file_hash = current_file_hash

                    st.success(f"✅ {filename} processed successfully!")
                    st.rerun()
                else:
                    st.error("Failed to process PDF document")

    # Display current document info
    if st.session_state.current_pdf:
        st.markdown("---")
        st.header("📋 Current Document")
        st.success(f"**Loaded:** {st.session_state.current_pdf}")
        if st.session_state.uploaded_file:
            st.info(f"**Size:** {len(st.session_state.uploaded_file.getvalue()) / 1024:.1f} KB")

        # Detect document type based on content
        doc_type = detect_document_type(st.session_state.vectorstore, st.session_state.current_pdf)
        st.info(f"**Document Type:** {doc_type}")

    st.markdown("---")
    st.header("ℹ️ About This Chatbot")
    st.info("""
    Upload any PDF document and ask questions about its content!
    - Supports all types of PDFs
    - Maintains conversation context
    - Provides source-based answers
    """)

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Clear Current Document"):
        st.session_state.vectorstore = None
        st.session_state.current_pdf = None
        st.session_state.uploaded_file = None
        st.session_state.file_hash = None
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if st.session_state.current_pdf and st.session_state.vectorstore:
    st.info(f"📄 Currently analyzing: **{st.session_state.current_pdf}**")

    prompt = st.chat_input(f'Ask about {st.session_state.current_pdf}...')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            with st.spinner("🔍 Searching document and generating response..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={'k': 4})
                relevant_docs = retriever.get_relevant_documents(prompt)

                # Debug: Show the retrieved context
                with st.expander("🔍 Debug: Retrieved Context"):
                    st.write(f"**Retrieved Content for Query:** {prompt}")
                    for i, doc in enumerate(relevant_docs):
                        st.write(f"**Chunk {i+1}:**")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

                if not relevant_docs:
                    response = "I couldn't find relevant information in this document to answer your question."
                else:
                    # Combine context from relevant documents
                    context = "\n\n".join([f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])

                    # Get response from Gemini
                    response = get_gemini_response(prompt, context, st.session_state.current_pdf)

                st.chat_message('assistant').markdown(response)

                # Show source information
                if relevant_docs:
                    with st.expander("📎 Source Information"):
                        st.write(f"Found {len(relevant_docs)} relevant sections from the document")
                        for i, doc in enumerate(relevant_docs):
                            st.write(f"**Source {i+1}:**")
                            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            st.text(preview)
                            st.write("---")

                st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

elif st.session_state.current_pdf and not st.session_state.vectorstore:
    st.warning("⚠️ Document selected but not processed. Please click 'Process Document'.")

else:
    st.info("👆 Please upload a PDF document using the sidebar to start chatting!")
    st.write("### How to use:")
    st.write("1. 📁 Upload a PDF file using the sidebar")
    st.write("2. 🚀 Click 'Process Document'")
    st.write("3. 💬 Start asking questions about your document!")
