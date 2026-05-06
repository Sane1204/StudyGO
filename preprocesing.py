from llama_index.core import VectorStoreIndex,Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
import json
import logging
from typing import Dict, List, Any, Optional,Union

logger=logging.getLogger(__name__)


def splitdata(Data: Union[List[Document], Document]):

    try:
        if isinstance(Data,Document):
            Data=[Data]
        text_splitter=SentenceSplitter(chunk_size=200,chunk_overlap=30)
        split=text_splitter.get_nodes_from_documents(Data)
        
        logger.info(f"Created {len(split)} successfully")
        
        return split
    except Exception as e:
        logger.error(f"Splitting failed: {e}")
        return []

def create_metadata(split,document_type,file_name):
    try:
          for node in split:
              node.metadata["document_type"]=document_type
              node.metadata["file_name"]=file_name
              

          return split
    except Exception as e:
        logger.error(f"error{e}")
        return []


def Embedding(split,Storage_context=None)->Optional[VectorStoreIndex]:

    try:
        Embedding_model=OllamaEmbedding(model_name="nomic-embed-text")
    
        index=VectorStoreIndex(
            embed_model=Embedding_model,
            nodes=split,
            storage_context=Storage_context,
            show_progress=True,
        )
        logger.info("embedding done successfully")
        return index 
    except Exception as e:
        logger.error(f"Embedding was unsuccesfull error was {e}")
        return None
    
def Embedding_for_mock(split)->Optional[VectorStoreIndex]:

    try:
        Embedding_model=OllamaEmbedding(model_name="nomic-embed-text")
    
        index_for_mock=VectorStoreIndex(
            embed_model=Embedding_model,
            nodes=split,
            show_progress=True,
        )
        logger.info("embedding done successfully")
        return index_for_mock
    except Exception as e:
        logger.error(f"Embedding was unsuccesfull error was {e}")
        return None
    
    
def Verify_embedding(index:VectorStoreIndex)->bool:
    try:
        vector_store=index._storage_context.vector_store
        node_ids=list(index.index_struct.nodes_dict.keys())
        missing_embedding=False
        for node_id in node_ids:
            embedding=vector_store.get(node_id)
            if embedding is None:
                logger.warning(f"Node id {node_id} has no embedding")
                missing_embedding=True
            else:
                logger.info(f"embedded succesfully in {node_id}")
        
        if missing_embedding:
            logger.warning("Some node embedding are missing")
            return False
        else:
            logger.info("All nodes are valid")
            return True
        
        
        
    except Exception as e:
        logger.error(f"error in verifying embedding")
        return False

            


        
