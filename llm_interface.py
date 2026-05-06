from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import logging
from typing import Optional

logger=logging.getLogger(__name__)

_current_model_name:str ='llama3'
fact_llm:Optional[Ollama]=None
q_and_a:Optional[Ollama]=None
doubt_sess:Optional[Ollama]=None
mock:Optional[Ollama]=None

def create_embedding()->OllamaEmbedding:
    return OllamaEmbedding(
        model_name="nomic-embed-text",
        request_timeout=180,
    )

def create_model(
        model_name:str,
        temperature: float=0.5,
        max_new_tokens:int=500,
        decoding_method: str ="sample",
        top_k:int=40,
        top_p:float=0.9,
)->Ollama:
     additional_params = {
        "decoding_method": decoding_method,
        "min_new_tokens": 250,
        "top_k": top_k,
        "top_p": top_p,
    }
     
     model=Ollama(
          model=model_name,
          temperature=temperature,
          max_new_tokens=max_new_tokens,
          request_timeout=1000,
          additional_kwargs=additional_params,
     )

     logger.info(f"CREATED {model_name} successfully")
     return model

def init_if_llm_is_needed():
     global fact_llm,q_and_a,doubt_sess,mock

     if fact_llm is None or q_and_a is None or doubt_sess is None or mock is None:
          
        fact_llm=create_model(
               model_name=_current_model_name,
               decoding_method="sample",
               temperature=0.3,
               top_k=40,
               top_p=0.9,
          )
        q_and_a=create_model(
                model_name=_current_model_name,
               decoding_method="sample",
               temperature=0.8,
               top_k=40,
               top_p=0.93,
          )
        doubt_sess=create_model(
               model_name=_current_model_name,
          
               decoding_method="sample",
               temperature=0.1,
               top_k=40,
               top_p=0.95,
          )
        mock=create_model(
               model_name=_current_model_name,
               decoding_method="sample",
               temperature=0.6,
               top_k=40,
               top_p=0.93,
          )
        logger.info("initialised all the llms")

def get_revision_llm()->Ollama:
     init_if_llm_is_needed()
     return fact_llm
def get_pratise_llm()->Ollama:
    init_if_llm_is_needed()
    return q_and_a
def get_doubt_llm()->Ollama:
    init_if_llm_is_needed()
    return doubt_sess
def get_mock_llm()->Ollama:
    init_if_llm_is_needed()
    return mock


def model_switching(new_model_id: str)-> None:
    global _current_model_name,q_and_a,mock,fact_llm,doubt_sess
    _current_model_name=new_model_id
    fact_llm=create_model(
               model_name=_current_model_name,
               decoding_method="sample",
               temperature=0.3,
               top_k=40,
               top_p=0.9,
          )
    q_and_a=create_model(
                model_name=_current_model_name,
               decoding_method="sample",
               temperature=0.8,
               top_k=40,
               top_p=0.93,
          )
    doubt_sess=create_model(
               model_name=_current_model_name,
               decoding_method="sample",
               temperature=0.1,
               top_k=40,
               top_p=0.95,
          )
    mock=create_model(
               model_name=_current_model_name,
               decoding_method="sample",
               temperature=0.6,
               top_k=40,
               top_p=0.93,
          )
    logger.info(f"Switched both LLM profiles to {new_model_id}")