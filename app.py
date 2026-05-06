import os
import sys
import logging
import tempfile
import streamlit as st
from ingest import extract_file
from preprocesing import splitdata,Embedding,Verify_embedding,create_metadata
from query_engine import q_and_a_engine,revison_notes_engine,mock_engine,doubt_engine
from prompt_template import REVISION_TEMPLATE,MOCKTEMPLATE,DOUBTTEMPLATE,QANDATEMPLATE
from llama_index.core import StorageContext,load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from ingest import extract_file
from preprocesing import splitdata,Embedding,Verify_embedding
from query_engine import q_and_a_engine,revison_notes_engine,mock_engine,doubt_engine
from llm_interface import model_switching

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)




st.set_page_config(
    page_title="STUDYGO",
    layout="wide",
    initial_sidebar_state="expanded"
        )
st.markdown("""
<style>
    /* Import a nice font from Google */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
 
    /* Style the entire app background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        font-family: 'Space Grotesk', sans-serif;
        color: #e0e0e0;
    }
 
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
 
    /* Style all buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }
 
    /* Style the text areas and inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.08);
        color: #4F8BF9;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
    }
 
    /* Style selectbox */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.08);
        color: #ffffff;
        border-radius: 8px;
    }
 
    /* Style the output text boxes */
    .output-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.7;
        white-space: pre-wrap;  
        color: #ffffff;      
    }
 
    /* Chat message styling */
    .chat-user {
        background: rgba(102,126,234,0.25);
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        text-align: right;
    }
    .chat-bot {
        background: rgba(255,255,255,0.07);
        border-radius: 12px 12px 12px 2px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
 
    /* Section headers */
    h1, h2, h3 {
        color: #c9b8ff 
    }
 
    /* Success/warning/error boxes */
    .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] ol,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em {
    color: #ffffff

    }
    [data-testid="stRadio"] label p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

if "index" not in st.session_state:
    st.session_state.index=None

if "nodes" not in st.session_state:
    st.session_state.nodes=None


if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]


if "file_processed" not in st.session_state:
    st.session_state.file_processed=False

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name=""
if "output" not in st.session_state:
    st.session_state.output = ""

@st.cache_resource

def load_backend_modules():
    try:
        sys.path.insert(0,"/mnt/user-data/uploads")

        import ingest
        import preprocesing
        import query_engine
        import llm_interface

        return ingest,preprocesing,query_engine,llm_interface,True,""
    except Exception as e:
        return None, None, None, None, False, str(e)

def save_uploaded_file(uploaded_file):
   
    suffix = os.path.splitext(uploaded_file.name)[1]   
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())           
        return tmp.name                                


def detect_document_type(file_name,document_type):

    file_name = file_name.lower()
    filename = os.path.basename(file_name)     
    filename = os.path.splitext(filename)[0]  
    filename = filename.replace(" ", "_")       
    Persist_dic=document_type+"_"+filename
    
    base_storage = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
    os.makedirs(base_storage, exist_ok=True)
    storage_path = os.path.join(base_storage, Persist_dic)
    return storage_path

def process_pdf(pdf,document_type,file_name):
    
    PERSIST_DIR=detect_document_type(file_name,document_type)
    
    try:
        
        index_exists=os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)

        if index_exists:
            with st.spinner("Loading your files"):
             logger.info("existing found")
             Storage_context=StorageContext.from_defaults(persist_dir=PERSIST_DIR)
             index=load_index_from_storage(Storage_context)
             docstore=Storage_context.docstore
             nodes = list(docstore.docs.values())
             return index,nodes
            
        else:
          with st.spinner("Progressing new"):
             logger.info("no existing index  found — creating new index")
             storage_context = StorageContext.from_defaults()
             extracting_pdf=extract_file(pdf)
             st.write(f"DEBUG extract: {type(extracting_pdf)}, len={len(extracting_pdf) if extracting_pdf else 0}")
            
             if not extracting_pdf:
                logger.error("failed to retrieve information")
                return None,None
            
             
             splitting_data=splitdata(extracting_pdf)
             st.write(f"DEBUG extract: {type(extracting_pdf)}, len={len(extracting_pdf) if extracting_pdf else 0}")
                
             metadata=create_metadata(splitting_data,document_type,file_name)

            
             vectoring_databse=Embedding(metadata)

             if not vectoring_databse:
              logger.error("error to embed data")
              return None,None
            
          
             verify_embed=Verify_embedding(vectoring_databse)
            
             if not verify_embed:
              logger.info("error in verification embedding or vector missing")
              return None,None
        

          vectoring_databse.storage_context.persist(persist_dir=PERSIST_DIR)
          logger.info("index persisted")
          return vectoring_databse ,metadata
    except Exception as e:
        st.error(f"PROCECESSING FAILED : {e}")
        logger.error(f"error occured {str(e)}")
        return None ,None
     
with st.sidebar:
    # st.markdown lets you write HTML directly — here we make a styled title
    st.markdown("<h2 style='color:#c9b8ff;'>📚 StudyGo</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0;font-style:italic;'>Your AI-powered study companion</p>", unsafe_allow_html=True)
    st.divider()

    model_name = st.selectbox(
        "🤖 Ollama Model",
        ["llama3", "qwen2.5"],
        help="Make sure this model is pulled in Ollama"  
    )
    if st.session_state.file_processed:
        st.warning("⚠️ Re-process your file to apply the new model.")

    mode = st.radio(
        "📌 Study Mode",
        options=["notes", "qanda", "doubt", "mock"],
        format_func=lambda x: {
            "notes": "📝 Revision Notes",
            "qanda": "❓ Practice Q&A",
            "doubt": "💬 Doubt Session",
            "mock":  "📋 Mock Exam"
        }[x]
    )

    st.divider()

    label = "📄 Upload Past Paper" if mode == "mock" else "📄 Upload Course Notes"
    uploaded_file = st.file_uploader(
        label,
        type=["pdf", "txt", "md", "docx", "pptx"],
        help="Supported: PDF, Word, PowerPoint, TXT, Markdown"
    )

    if st.button("Process file", disabled=(uploaded_file is None),):
        

      try:
         model_switching(model_name)
         document_type = "past_paper" if mode == "mock" else "notes"
         tmp_path = save_uploaded_file(uploaded_file)

         index, nodes = process_pdf(tmp_path, document_type, uploaded_file.name)

         if index is not None and nodes is not None:
          st.session_state.index = index
          st.session_state.nodes = nodes
          st.session_state.file_processed = True
          st.session_state.current_file_name = uploaded_file.name
          st.session_state.chat_history = []
          st.session_state.output = ""
          st.success(f"'{uploaded_file.name}' processed!")
          st.rerun()
         else:
          st.error("failed to process file")
      except Exception as e:
        st.error(f"Backend processing failed: {e}")
    

            
        

    st.divider()
    if st.session_state.file_processed:
     st.markdown(f"**Loaded:** `{st.session_state.current_file_name}`")
     st.markdown(f"**Model:** `{model_name}`")

     if st.button("🗑️ Clear & Reset"):
        st.session_state.index = None
        st.session_state.nodes = None
        st.session_state.chat_history = []
        st.session_state.output = ""
        st.session_state.file_processed = False
        st.session_state.current_file_name = ""
        st.rerun()
    else:
     st.info("⬆️ Upload a file and click Process to begin.")

st.markdown("StudyGO Ai Assistant")
st.markdown("Upload your course notes or past papers and let AI supercharge your revision.")
st.divider()


if not st.session_state.file_processed:
     col1,col2,col3,col4=st.columns(4)
     with col1:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05);border-radius:12px;padding:1rem;text-align:center'>
            <div style='font-size:2rem'>📝</div>
            <b>Revision Notes</b><br>
            <small>Auto-generate structured notes from any document</small>
        </div>
        """, unsafe_allow_html=True)
 
     with col2:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05);border-radius:12px;padding:1rem;text-align:center'>
            <div style='font-size:2rem'>❓</div>
            <b>Practice Q&A</b><br>
            <small>Get exam-style questions graded by difficulty</small>
        </div>
        """, unsafe_allow_html=True)
 
     with col3:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05);border-radius:12px;padding:1rem;text-align:center'>
            <div style='font-size:2rem'>💬</div>
            <b>Doubt Session</b><br>
            <small>Ask anything get clear explanations instantly</small>
        </div>
        """, unsafe_allow_html=True)
 
     with col4:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05);border-radius:12px;padding:1rem;text-align:center'>
            <div style='font-size:2rem'>📋</div>
            <b>Mock Exam</b><br>
            <small>Generate a full mock paper from past papers</small>
        </div>
        """, unsafe_allow_html=True)
 
     st.markdown("<br><br><center><i>👈 Upload a file from the sidebar to get started</i></center>", unsafe_allow_html=True)
     st.stop()

if mode == "notes":
    st.markdown("## 📝 Revision Notes Generator")
    st.markdown("Click below to generate structured revision notes from your uploaded document.")
 
    # Optional: let user customise the query
    # st.text_input = single-line text box. value= sets the default.
    custom_query = st.text_input(
        "🎯 Focus topic (optional)",
        value="key concepts definitions explanations",
        help="Leave as-is for general notes, or type a specific topic"
    )
 
    if st.button("📝 Generate Revision Notes"):
        with st.spinner("🧠 Generating notes... this may take a moment"):
            result = revison_notes_engine(
                st.session_state.index,
                st.session_state.nodes,
                custom_query
            )
            # Store output in session_state so it survives reruns
            st.session_state.output = str(result)
 
    # Show output if it exists
    # The output persists in session_state even after reruns
    if st.session_state.output:
        st.markdown("### 📄 Your Notes")
        # st.markdown renders the output with formatting (bold, bullets etc)
        st.markdown(st.session_state.output)
 
        # st.download_button lets the user download any text as a file
        st.download_button(
            label="⬇️ Download Notes as .txt",
            data=st.session_state.output,
            file_name=f"notes_{st.session_state.current_file_name}.txt",
            mime="text/plain"
        )
 
# ============================================================
# MODE: PRACTICE Q&A
# Very similar to notes mode — one-shot generation.
# ============================================================
elif mode == "qanda":
    st.markdown("## ❓ Practice Question Generator")
    st.markdown("Generate exam-style questions grouped by difficulty.")
 
    custom_query = st.text_input(
        "🎯 Focus topic (optional)",
        value="important examinable topics concepts",
    )
 
    if st.button("❓ Generate Practice Questions"):
        with st.spinner("🧠 Generating questions..."):
            result = q_and_a_engine(
                st.session_state.index,
                st.session_state.nodes,
                custom_query
            )
            st.session_state.output = str(result)
 
    if st.session_state.output:
        st.markdown("### 📋 Practice Questions")
        st.markdown(st.session_state.output)
 
        st.download_button(
            label="⬇️ Download Questions as .txt",
            data=st.session_state.output,
            file_name=f"questions_{st.session_state.current_file_name}.txt",
            mime="text/plain"
        )
 
# ============================================================
# MODE: DOUBT SESSION (CHATBOT)
#
# CONCEPT: How to build a chatbot in Streamlit
# 
# We store messages in st.session_state.chat_history as a list:
# [{"role": "user", "content": "..."}, {"role": "bot", "content": "..."}]
#
# On every rerun we LOOP through the history and display all messages.
# Then at the bottom, st.chat_input waits for a new message.
# When the user types and hits Enter:
# 1. We add their message to history
# 2. Call the AI
# 3. Add the AI reply to history
# 4. Streamlit reruns, loop displays everything including new messages
# ============================================================
elif mode == "doubt":
    st.markdown("## 💬 Doubt Solving Session")
    st.markdown("Ask questions about your material — get clear, step-by-step explanations.")
 
    # Create a container for the chat — keeps it scrollable
    chat_container = st.container()
 
    with chat_container:
        # Loop through ALL past messages and display them
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                # st.chat_message creates a styled message bubble with an avatar
                with st.chat_message("user", avatar="🧑‍🎓"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message["content"])
 
    # st.chat_input creates the message box at the BOTTOM of the screen
    # It returns None when empty, or the string when user submits
    user_input = st.chat_input("Ask a question about your material...")
 
    if user_input:
        # 1. Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
 
        # 2. Call the doubt engine with the user's question
        with st.spinner("🤔 Thinking..."):
            response = doubt_engine(
                st.session_state.index,
                st.session_state.nodes,
                user_input
            )
            bot_reply = str(response)
 
        # 3. Add bot reply to history
        st.session_state.chat_history.append({"role": "bot", "content": bot_reply})
 
        # 4. st.rerun() forces a rerun so the new messages appear immediately
        st.rerun()
 
    # Button to clear just the chat history
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
 
# ============================================================
# MODE: MOCK EXAM
# Same pattern as notes/qanda — one-shot generation.
# ============================================================
elif mode == "mock":
    st.markdown("## 📋 Mock Exam Generator")
    st.markdown("Generate a full mock exam paper based on your uploaded past paper.")
 
    # st.info shows a blue information box
    st.info("💡 Tip: Upload a past exam paper (not notes) for best results in thismode.")
  
    custom_query = st.text_input(
        "🎯 Instructions (optional)",
        value="make your own exam style mock paper from the content provided"
    )
 
    if st.button("📋 Generate Mock Exam"):
        with st.spinner("📝 Writing your mock exam..."):
            result = mock_engine(
                st.session_state.index,
                st.session_state.nodes,
                custom_query
            )
            st.session_state.output = str(result)
 
    if st.session_state.output:
        st.markdown("### 📄 Your Mock Exam")
        st.markdown(st.session_state.output)
        st.download_button(
            label="⬇️ Download Mock Exam as .txt",
            data=st.session_state.output,
            file_name=f"mock_exam_{st.session_state.current_file_name}.txt",
            mime="text/plain"
        )
 
# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<center><small>StudyGo • Powered by LlamaIndex + Ollama • Running locally 🔒</small></center>",
    unsafe_allow_html=True
)
 
    


