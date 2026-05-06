import sys
import time
import logging
import argparse
import os

from ingest import extract_file
from preprocesing import splitdata,Embedding,Verify_embedding,create_metadata
from query_engine import q_and_a_engine,revison_notes_engine,mock_engine,doubt_engine
from prompt_template import REVISION_TEMPLATE,MOCKTEMPLATE,DOUBTTEMPLATE,QANDATEMPLATE
from llama_index.core import StorageContext,load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm=Ollama(model="llama3")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def detect_document_type(file_name,document_type):

    file_name = file_name.lower()
    filename = os.path.basename(file_name)      # Remove file path
    filename = os.path.splitext(filename)[0]   # Remove .pdf extension
    filename = filename.replace(" ", "_")       # Replace spaces with _
    Persist_dic=document_type+"_"+filename
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    storage_path = os.path.join(base_dir, "storage", Persist_dic)
    return storage_path

def process_pdf(pdf,document_type,file_name):
    
    PERSIST_DIR=detect_document_type(file_name,document_type)
    
    try:
        index_exists=os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)

        if index_exists:
            logger.info("existing found")
            Storage_context=StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index=load_index_from_storage(Storage_context)
            docstore=Storage_context.docstore
            nodes = list(docstore.docs.values())
            return index,nodes
            
        else:
            
            logger.info("no existing index  found — creating new index")
            storage_context = StorageContext.from_defaults()
            extracting_pdf=extract_file(pdf)
            
            if not extracting_pdf:
                logger.error("failed to retrieve information")
                return None
        
            splitting_data=splitdata(extracting_pdf)

            metadata=create_metadata(splitting_data,document_type,file_name)
            vectoring_databse=Embedding(metadata)

            if not vectoring_databse:
              logger.error("error to embed data")
              return None
        
            verify_embed=Verify_embedding(vectoring_databse)
            if not verify_embed:
              logger.info("error in verification embedding or vector missing")
              return None
        

            vectoring_databse.storage_context.persist(persist_dir=PERSIST_DIR)
            logger.info("index persisted")
        

        
            return vectoring_databse ,metadata
    except Exception as e:
        logger.error(f"error occured {str(e)}")
        return

def chatbot_interface(index,nodes,user_query):
    print("\nYou can now ask more in-depth questions about this person. Type 'exit', 'quit', or 'bye' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        
        print("Bot is typing...", end='')
        sys.stdout.flush()
        time.sleep(1)  
        print('\r', end='')
        
        response = doubt_engine(index,nodes,user_query)
        
        try:
            text = response.response
            
        except Exception:
            text = str(response)
            print(f"Bot: {text.strip()}\n")
            

def main():
    print("🚀 main() started")
    parser=argparse.ArgumentParser(description="Studygo")
    parser.add_argument('--mode',type=str,choices=['notes','doubt','qanda','mock'],default="notes")
    parser.add_argument('--notes', type=str, help='path to file')
    parser.add_argument('--qanda', type=str, help='path to file')
    parser.add_argument('--mock', type=str, help='path to past paper ')
    parser.add_argument('--doubt', action="store_true", help='solving your doubt ')
    parser.add_argument('--model', type=str, help="Model name for Ollama (optional)")
    args=parser.parse_args()

    if args.model:
        from llm_interface import model_switching
        model_switching(args.model)

    notes_index=None
    past_paper=None
    

    if args.mode in ["notes", "qanda", "doubt"]:
        if args.mode == "notes":
            query = "key concepts definitions explanations "
        elif args.mode == "qanda":
            query = "important examinable topics concepts"
        elif args.mode == "doubt":
            query = "question:"

        # These modes need the course notes
        notes = args.notes or input("Enter course notes file path: ").strip()
        document_type="notes"
        file_name=notes
        notes_index,nodes = process_pdf(notes,document_type,file_name)
        if not notes_index:
            return 
        notes_index
    
        if args.mode == "notes":
            
            out = revison_notes_engine(notes_index,nodes,query)

            print("\n=== NOTES ===")
            print(out)

        elif args.mode == "qanda":
        
            # Either generate a set of practice questions OR start a QA loop depending on your engine
            out = q_and_a_engine(notes_index,nodes,query)
              # if your engine generates a set
            print("\n=== PRACTICE Q&A ===")
            print(out)

        elif args.mode == "doubt":
            
            chatbot_interface(notes_index,nodes,query)

    elif args.mode == "mock":
        # Mock mode needs the past paper
        mock = args.mock or input("Enter past paper file path: ").strip()
        file_name=mock
        query = "make you own exam style mock paper from the content provided"
        document_type="past_paper"
        past_paper,nodes = process_pdf(mock,document_type,file_name)
        if not past_paper:
            return

        out = mock_engine(past_paper,nodes,query)
        print("\n=== MOCK EXAM ===")
        print(out)

if __name__ == "__main__":
    main()