from llama_index.retrievers.bm25 import BM25Retriever
from query_engine import hybrid_retriever,join_text
import prompt_template
from llm_interface import get_doubt_llm
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from llama_index.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    AnswerSimilarity,
)
from main import process_pdf
def run_doubt_with_ref(index,nodes,query,top_k=5):
    llm=get_doubt_llm()

    bm25=BM25Retriever.from_defaults(nodes=nodes)
    vector_retriever=index.as_retriever()

    retrieved_nodes=hybrid_retriever(
        vector_retriever,
        bm25,
        query,
        top_k=top_k
    
    )
    
    retrieved_context=[nodes.get_content() for nodes in retrieved_nodes]
    context=join_text(retrieved_nodes)

    final_prompt=(prompt_template.GENERAL_TEMPLATE+
                  prompt_template.DOUBTTEMPLATE
                  ).format(context_str=context,
                           query_str=query
     )
    response=llm.complete(final_prompt)

    return{
        "user_input": query,
        "retrieved_contexts":retrieved_context,
        "response":str(response),
        "nodes":retrieved_nodes,
    }


def build_predictions(index,nodes):
    df=pd.read_csv("questions.csv")

   
  
    query=[]
    for _, rows in df.iterrows():
        result=run_doubt_with_ref(index, nodes, rows["user_input"])

        

        query.append({
            "user_input":rows["user_input"],
            "retrieved_contexts":result["retrieved_contexts"],
            "response":result["response"],
            "reference":rows["reference"],
        })
    return pd.DataFrame(query)
    
def main():

    index,nodes=process_pdf(pdf="ai_intro.pdf",
                            document_type="notes",
                            file_name="ai_intro.pdf",)
    
    prediction=build_predictions(index,nodes)

    evaluation=Dataset.from_pandas(prediction)
    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",  
        temperature=0
)

    result = evaluate(
    dataset=evaluation,
    metrics=[
        Faithfulness(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        AnswerSimilarity(),
    ],
    llm=evaluator_llm
)
    
    result=result.to_pandas()
    print(result)
    result.to_csv("ragas_result.csv",index=False)

    

if __name__=="__main__":
    main()


    
