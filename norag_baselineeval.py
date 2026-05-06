from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import pandas as pd
import nltk

nltk.download("punkt")

Settings.embed_model=OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm=Ollama(model="llama3")

NO_RAG_PROMPT="""You are academic assitant.
Answer the following question with your knowledge

Question: {query}

Answer:
"""

def get_no_rag_answe(llm,query):
    prompt=NO_RAG_PROMPT.format(query=query)
    response=llm.complete(prompt)
    return str(response)

def compute_score(refrence,response):
    scorer=rouge_scorer.RougeScorer(['rouge1','rougeL'],use_stemmer=True)
    rouge=scorer.score(refrence,response)


    ref_tokens=[refrence.split()]
    hyp_tokens=response.split()
    bleu= sentence_bleu(
        ref_tokens,hyp_tokens,
        smoothing_function=SmoothingFunction().method1
    )

    return{"rouge1":rouge['rouge1'].fmeasure,
           "rougeL":rouge['rougeL'].fmeasure,
            "bleu":bleu,}

def main():
    llm=Ollama(model="llama3",temperatue=0.1,request_timeout=1000)
    df=pd.read_csv("questions.csv")


    results=[]
    responses=[]
    references=[]


    for _,row in df.iterrows():
        raw=row['reference']
        if pd.isna(raw) or str(raw).strip() =="":
            continue

        query=row["user_input"]
        reference=row["reference"]


        print(f"Running: {query[:60]}...")
        response = get_no_rag_answe(llm, query)

        scores = compute_score(reference, response)
        scores["user_input"] = query
        scores["response"] = response
        scores["reference"] = reference
        results.append(scores)

        responses.append(response)
        references.append(reference)

    
    P, R, F1 = bert_score(responses, references, lang="en")
    for i, row in enumerate(results):
        row["bertscore_f1"] = F1[i].item()

    results_df = pd.DataFrame(results)

    print("\n==== NO-RAG BASELINE RESULTS ====")
    print(f"ROUGE-1:    {results_df['rouge1'].mean():.3f}")
    print(f"ROUGE-L:    {results_df['rougeL'].mean():.3f}")
    print(f"BLEU:       {results_df['bleu'].mean():.3f}")
    print(f"BERTScore:  {results_df['bertscore_f1'].mean():.3f}")

    results_df.to_csv("no_rag_baseline_results.csv", index=False)
    print("\nSaved to no_rag_baseline_results.csv")

if __name__ == "__main__":
    main()  

    