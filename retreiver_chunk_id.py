from main import process_pdf
from llama_index.retrievers.bm25 import BM25Retriever
from query_engine import hybrid_retriever
import pandas as pd

index,nodes= process_pdf(
    pdf="ai_intro.pdf",
    document_type="notes",
    file_name="ai_intro.pdf"
)
df = pd.read_csv("questions.csv")
questions=[]
for _, rows in df.iterrows():
    result=rows["user_input"]

    questions.append(result)


for query in questions:
 vector_retriever=index.as_retriever()
 bm25=BM25Retriever.from_defaults(nodes=nodes)
 hybrid=hybrid_retriever(vector_retriever,bm25,query,top_k=5)
 
 print(f"QUESIONS:{query}")
 print("="*60)
 for i,node in enumerate(hybrid):
    print(f"CHUNK: {i+1}")
    print(f"id: {node.node_id}")
    print(f"content: {node.get_content()[:400]} ")
    print("-"*40)
