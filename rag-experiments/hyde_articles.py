import os
import pandas as pd
import bm25s
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline


def main():
    # Read in and look at text column of data csv
    docs = pd.read_csv('') # add data file

    corpus = []
    lens = []

    for index, row in docs.iterrows():
        corpus.append(row['Text'])
        lens.append(1)

    queries = list(docs['Claim'])
    K_s = [1,3,5,10]
    
    token = "" # add huggingFace token

    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))

    du_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', token=token)
    du_passages = du_model.encode(corpus, show_progress_bar=True)

    # Define model
    model_id = "microsoft/phi-4"
    
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token
    )

    for K in K_s:
        hybrid_du_all = []
        for query_old in queries:
            # Define prompt
            prompt = 'Leg in een paar zinnen uit waarom onderstaande claim niet klopt.\n\n' + query_old

            messages = [
                {"role": "system", "content": "Je bent een behulpzame assistent die misinformatie corrigeert."},
                {"role": "user", "content": prompt},
            ]

            outputs = pipe(
                messages,
                max_new_tokens=256,
            )

            query = outputs[0]["generated_text"][-1]['content']

            # BM25s
            bm25_results, bm25_scores = retriever.retrieve(bm25s.tokenize(query), k=100)

            # ST-DU
            query_embedding = du_model.encode(query)
            similarity = du_model.similarity(query_embedding, du_passages)

            # Zip similarities and corresponding documents
            du_zipped = list(zip(corpus, list(similarity[0])))
            du_zipped = sorted(du_zipped, key=lambda x: x[1], reverse=True)

            # zip hybrids
            bm25_results = bm25_results[0]
            bm25_scores = bm25_scores[0]
            bm25 = list(zip(bm25_results, bm25_scores))

            # HYBRID-DU
            hybrid_du = []
            for item in bm25:
                du_dict = dict(du_zipped)
                du_score = float(du_dict[item[0]])
                score = 0.3 * item[1] + du_score
                hybrid_du.append((item[0], score))

            hybrid_du_sorted = sorted(hybrid_du, key=lambda tup: tup[1], reverse=True)[:K]

            hybrid_du_relevant = 0
            # Check if each item in the retrieved docs is in the query's corresponding doc
            for item in hybrid_du_sorted:
                if docs[docs['Claim'] == query_old].shape[0] == 1:
                    if item[0] in docs[docs['Claim'] == query_old].iloc[0].get("Text"):
                        hybrid_du_relevant += 1
                else:
                    for i, row in docs[docs['Claim'] == query_old].iterrows():
                        if item[0] in row.get("Text"):
                            hybrid_du_relevant += 1

            # Calculate the total number of relevant docs
            if docs[docs['Claim'] == query_old].shape[0] == 1:
                    total_relevant = lens[docs[docs['Claim'] == query_old].index[0]]
            else:
                total_relevant = 0
                for i, row in docs[docs['Claim'] == query_old].iterrows():
                    total_relevant += lens[int(i)]

            hybrid_du_r_k = hybrid_du_relevant / K
            hybrid_du_all.append(hybrid_du_r_k)
        print(f'average P@{K} for HyDE-DU = {sum(hybrid_du_all) / len(hybrid_du_all)}')
        with open('hyde_articles.txt', 'a') as f:
                f.write(f'average P@{K} for HyDE-DU = {sum(hybrid_du_all) / len(hybrid_du_all)}\n')

main()
