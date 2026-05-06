import logging
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.llms.ollama import Ollama
import prompt_template
from llm_interface import create_model,get_doubt_llm,get_mock_llm,get_pratise_llm,get_revision_llm
from llama_index.retrievers.bm25 import BM25Retriever
from typing import List
from preprocesing import splitdata,Embedding
logger=logging.getLogger(__name__)

storage_context=StorageContext.from_defaults()


def join_text(retrieve_nodes):
    texts = []

    for result in retrieve_nodes:
        node = getattr(result, "node", result)

        text = node.get_content()
        metadata = getattr(node, "metadata", {})

        source = metadata.get("file_name", "Unknown")
        page = metadata.get("page_label", "Unknown")
        document_type=metadata.get("document_type","Unknown")


        formatted_chunk = f"[Source: {source}, Page: {page}, Document_type: {document_type}\n{text}]"
        texts.append(formatted_chunk)

    return "\n\n".join(texts)

def tageted_nodes(nodes,document_type):
  filter_node=[]
  target=document_type
  for node in nodes:
    if node.metadata["document_type"]== target:
      filter_node.append(node)
  return filter_node

def hybrid_retriever(vector_retriever,bm_25_retriever, query,top_k=5):

  bm25_result=bm_25_retriever.retrieve(query)
  vector_result= vector_retriever.retrieve(query)

  k=60
  merge_scores={}
  

  for rank,node in enumerate(vector_result):
    node_id=node.node.node_id
    node_content=node.node
    if node_id not in merge_scores:
      merge_scores[node_id]={
        'node':node_content,
        'rrf_score':0,
        'rank':[]
        }
    rrf_contribution=1.0/(rank+1 + k)
    merge_scores[node_id]['rrf_score']+= rrf_contribution
    merge_scores[node_id]['rank'].append((rank+1))
  
  for rank,node in enumerate(bm25_result):
    node_content=node.node
    node_id=node.node.node_id
    if node_id not in merge_scores:
       merge_scores[node_id]={
        'node':node_content,
        'rrf_score':0,
        'rank':[]
        }
    rrf_contribution=1.0/(rank+1 + k)
    merge_scores[node_id]['rrf_score']+= rrf_contribution
    merge_scores[node_id]['rank'].append((rank+1))


  sorted_result=sorted(
    merge_scores.values(),
    key=lambda x:x['rrf_score'],
    reverse=True
  )
  hybrid_result=[]
  for result in sorted_result:
    result= result['node']
    hybrid_result.append(result)
  return hybrid_result[:top_k]





  

  # # for result in vector_result:
  # #   nodes=getattr(result,"node",result)
  # #   node_id=getattr(nodes,"node_id",None)

  # #   key = node_id if node_id else nodes.get_content()[:200]

  # #   if key not in seen:
  # #     seen.add(key)
  # #     merge.append(result)
  
  # # for result in bm25_result:
  # #   nodes=getattr(result,"node",result)
  # #   node_id=getattr(nodes,"node_id",None)

  # #   key = node_id if node_id else nodes.get_content()[:200]

  # #   if key not in seen:
  # #     seen.add(key)
  # #     merge.append(result)
  # # print(top_k)
  # # return merge[:top_k]

  # #normalise
  # max_vector_store=max([s.score for s in vector_result]) if vector_result else 1 
  # if max_vector_store == 0:
  #   print("error")

  # for result in vector_result:
  #   key=result.text.strip()
  #   score=result.score/max_vector_store
  #   vector_set[key]=score
  #   merge[key]=result
  #   print(merge[key])
  # #normalise
  # max_bm25_store=max([s.score for s in bm25_result]) if bm25_result else 1
  # if max_bm25_store == 0:
  #  print("error")
  # for result in bm25_result:
  #   key=result.text.strip()
  #   score=result.score/max_bm25_store
  #   bm_25_set[key]=score
  #   merge[key]=result
  #   print(merge[key])

  # hybrid_result=[]
  # for key in merge:
  #   vector_score=vector_set.get(key)
  #   bm25_score=bm_25_set.get(key)
  #   hybrid_score=0.65*vector_score+0.35*bm25_score
    
  # hybrid_result.append({
  #           'node': merge[key],
  #           'vector_score': vector_score,
  #           'bm25_score': bm25_score,
  #           'hybrid_score': hybrid_score
  #       })
    
    
  # hybrid_result.sort(key=lambda x: x['hybrid_score'], reverse=True)
  # return hybrid_result[:top_k]
  
def revison_notes_engine(index:VectorStoreIndex,filter_node,query:str)->str:
 try:    
    llm=get_revision_llm()
   
    bm_25_retriever=BM25Retriever.from_defaults(nodes=filter_node)
    
    vector_retreiver=index.as_retriever()
    
   
    retrieve_nodes=hybrid_retriever(vector_retreiver,bm_25_retriever,query,top_k=5)

    context_str=join_text(retrieve_nodes)
    
    revision_template=prompt_template.GENERAL_TEMPLATE+prompt_template.REVISION_TEMPLATE
    ##final_string=PromptTemplate.format(revision_template,query,context_str)
    final_string=revision_template.format(context_str=context_str,query_str=query)
  
    answer=llm.complete(final_string)
  
    return answer
 except Exception as e:
   logger.error(f"error generating revision notes: {e}")
   return "Failed to prepare notes here "
 

def q_and_a_engine(index:VectorStoreIndex,filter_node,query:str)->str:
  try:
    llm=get_pratise_llm()
   


    bm_25_retriever=BM25Retriever.from_defaults(nodes=filter_node)
    
    vector_retreiver=index.as_retriever()
   
    retrieve_nodes=hybrid_retriever(vector_retreiver,bm_25_retriever,query,top_k=5)

    context_str=join_text(retrieve_nodes)
    
    revision_template=prompt_template.GENERAL_TEMPLATE+prompt_template.QANDATEMPLATE
    
    final_string=revision_template.format(context_str=context_str,query_str=query)
  
    answer=llm.complete(final_string)
  
    return answer
  
  except Exception as e:
    logger.error(f"failed to generate qanda :{e}")
    return "not able to generate q and a"

def doubt_engine(index:VectorStoreIndex, filter_node,query:str)->str:
  try:
    llm=get_doubt_llm() 
  

    bm_25_retriever=BM25Retriever.from_defaults(nodes=filter_node)
    
    vector_retreiver=index.as_retriever()
   
    retrieve_nodes=hybrid_retriever(vector_retreiver,bm_25_retriever,query,top_k=5)

    context_str=join_text(retrieve_nodes)
    
    revision_template=prompt_template.GENERAL_TEMPLATE+prompt_template.DOUBTTEMPLATE
    
    final_string=revision_template.format(context_str=context_str,query_str=query)
  
    answer=llm.complete(final_string)
  
    return answer
  
  except Exception as e:
    logger.error(f"failed to generate your answer for this :{e}")
    return "not able to solve your doubt"
  
def mock_engine(index:VectorStoreIndex,filter_node,query:str)->str:
  try:
    llm=get_mock_llm()
   
    bm_25_retriever=BM25Retriever.from_defaults(nodes=filter_node)
    
    vector_retreiver=index.as_retriever()
   
    retrieve_nodes=hybrid_retriever(vector_retreiver,bm_25_retriever,query,top_k=5)

    context_str=join_text(retrieve_nodes)
    
    revision_template=prompt_template.MOCKTEMPLATE
    
    final_string=revision_template.format(context_str=context_str,query_str=query)
  
    answer=llm.complete(final_string)
  
    return answer
  
  except Exception as e:
    logger.error(f"failed to generate your mock for this :{e}")
    return "not able to generate mock for this"