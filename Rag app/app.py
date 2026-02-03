import streamlit as st
import importlib.util
import os


# Page configuration
st.set_page_config(
    page_title="Movie RAG AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #e8f4f8;
        margin-right: 20%;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎬 Movie RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about movies using AI + your dataset</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for status and info
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    
    google_api_key = st.text_input("Gemini API Key", type="password", help="Get your key at https://aistudio.google.com/app/apikey")
    model_options = ["models/gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "Custom"]
    selected_model_option = st.selectbox("Select Model", model_options, index=0)
    
    if selected_model_option == "Custom":
        model_name = st.text_input("Enter Model Name", value="gemini-1.5-flash")
    else:
        model_name = selected_model_option
    
    if not google_api_key:
        st.warning("⚠️ Please enter your Gemini API Key to proceed.")
    else:
        st.success("✅ API Key provided")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Actions")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if google_api_key and st.button("🔍 List Available Models"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.code("\n".join(models))
        except Exception as e:
            st.error(f"Error listing models: {e}")
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This is a Movie RAG (Retrieval-Augmented Generation) assistant.
    
    **Features:**
    - Ask questions about movies
    - Get AI-powered answers
    - Uses your movie dataset
    
    **Tech Stack:**
    - Streamlit
    - LangChain
    - FAISS Vector DB
    - Google Gemini (LLM)
    """)

# Check if Ollama is running (main check)
if not google_api_key:
    st.info("👈 **Please enter your Google Gemini API Key in the sidebar to start.**")
    st.stop()

# Load RAG chain
@st.cache_resource
def load_qa_chain(api_key, model_name):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "04_rag_chain.py")
        spec = importlib.util.spec_from_file_location("rag_chain", file_path)
        rag_chain = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_chain)
        load_rag = rag_chain.load_rag
        qa = load_rag(api_key, model_name)
        return qa, None
    except Exception as e:
        return None, str(e)

qa, error = load_qa_chain(google_api_key, model_name)

if error:
    st.error(f"❌ **Error loading RAG chain:** {error}")
    st.stop()

# Display chat history
st.markdown("### 💬 Conversation")
if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("👋 **Welcome!** Start a conversation by asking a question about movies below.")

# Chat input
query = st.chat_input("Ask something about movies...")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                answer = qa.invoke({"query": query})
                response = answer["result"]
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
                
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    st.warning("⚠️ **Model Not Found**")
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=google_api_key)
                        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        
                        st.info(f"""
                        **The model '{model_name}' was not found.**
                        
                        Here are the models available for your API key:
                        """)
                        st.code("\n".join(available_models))
                        st.markdown("**Please select 'Custom' in the sidebar and enter one of the names above exactly.**")
                    except Exception as list_error:
                         st.error(f"Could not list available models: {list_error}")
