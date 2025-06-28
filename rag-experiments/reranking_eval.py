import os
import bm25s
from sentence_transformers import SentenceTransformer
import pandas as pd
from transformers import AutoModelForSequenceClassification


def rerank_results(query, documents, model):
    """rerank retrieved documents"""
    # construct sentence pairs and compute scores
    sentence_pairs = [[query, doc] for doc in documents]
    scores = model.compute_score(sentence_pairs, max_length=1024)

    # reorder scored documents
    zipped_scores = zip(documents, scores)
    reordered_docs_scores = sorted(zipped_scores, key=lambda tup: tup[1], reverse=True)
    reordered_docs = [item[0] for item in reordered_docs_scores]

    return reordered_docs


def main():
    # Read in and look at text column of data csv
    docs = pd.read_csv('') # add csv file
    docs.head()

    corpus = []
    lens = []

    # split into paragraphs
    for index, row in docs.iterrows():
        text = row['Text'].replace('\r', '\n')
        text = text.replace('\n\n\n\n', '\n')
        text = text.replace('\n\n\n', '\n')
        text = text.replace('\n ', '\n')
        text_list = text.split('\n')
        this_len = 0
        for item in text_list:
            if item.count('.') < 2 and item != '':
                item = corpus[-1] + ' ' + item
                corpus[-1] = item
            elif item != '':
                corpus.append(item)
                this_len += 1
        lens.append(this_len)

    queries = list(docs['Claim'])
    K_s = [1,3,5,10]

    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))

    token = '' # add token
    du_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', token=token)
    du_passages = du_model.encode(corpus, show_progress_bar=True)

    for K in K_s:
        bm25_all = []
        st_du_all = []
        hybrid_du_all = []
        hybrid_du_reranked = []

        for query in queries:
            #BM25s
            bm25_results, bm25_scores = retriever.retrieve(bm25s.tokenize(query), k=100)

            #ST-P
            query_embedding = du_model.encode(query)
            similarity = du_model.similarity(query_embedding, du_passages)

            # Zip similarities and corresponding documents
            du_zipped = list(zip(corpus, list(similarity[0])))
            du_zipped = sorted(du_zipped, key=lambda x: x[1], reverse=True)
            st_du_results = [du_zipped[i][0] for i in range(K)]

            # zip hybrids
            bm25_results = bm25_results[0]
            bm25_scores = bm25_scores[0]
            bm25 = list(zip(bm25_results, bm25_scores))

            #HYBRID-DU
            hybrid_du = []
            for item in bm25:
                du_dict = dict(du_zipped)
                du_score = float(du_dict[item[0]])
                score = 0.3 * item[1] + du_score
                hybrid_du.append((item[0], score))
            
            hybrid_du_sorted = sorted(hybrid_du, key=lambda tup: tup[1], reverse=True)[:K]
            hybrid_4_reranking = sorted(hybrid_du, key=lambda tup: tup[1], reverse=True) # The full list of 100 for reranking

            # Scoring
            bm25_relevant = 0
            for item in bm25_results[:K]:
                if docs[docs['Claim'] == query].shape[0] == 1:
                    if item in docs[docs['Claim'] == query].iloc[0].get("Text"):
                        bm25_relevant += 1
                else:
                    for i, row in docs[docs['Claim'] == query].iterrows():
                        if item in row.get("Text"):
                            bm25_relevant += 1

            bm25_r_k = bm25_relevant / K

            # ST DU
            st_du_relevant = 0
            for item in st_du_results:
                if docs[docs['Claim'] == query].shape[0] == 1:
                    if item in docs[docs['Claim'] == query].iloc[0].get("Text"):
                        st_du_relevant += 1
                else:
                    for i, row in docs[docs['Claim'] == query].iterrows():
                        if item in row.get("Text"):
                            st_du_relevant += 1

            st_du_r_k = st_du_relevant / K
            
            # Hybrid DU
            hybrid_du_relevant = 0
            for item in hybrid_du_sorted:
                if docs[docs['Claim'] == query].shape[0] == 1:
                    if item[0] in docs[docs['Claim'] == query].iloc[0].get("Text"):
                        hybrid_du_relevant += 1
                else:
                    for i, row in docs[docs['Claim'] == query].iterrows():
                        if item[0] in row.get("Text"):
                            hybrid_du_relevant += 1

            hybrid_du_r_k = hybrid_du_relevant / K

            # Reranked
            reranked_relevant = 0
            documents = [item[0] for item in hybrid_4_reranking]
            rerank_model =  AutoModelForSequenceClassification.from_pretrained(
                'jinaai/jina-reranker-v2-base-multilingual',
                torch_dtype="auto",
                trust_remote_code=True,
            )

            rerank_model.to('cuda') # or 'cpu' if no GPU is available
            rerank_model.eval()

            reranked_docs = rerank_results(query, documents, rerank_model)

            for item in reranked_docs[:K]:
                if docs[docs['Claim'] == query].shape[0] == 1:
                    if item in docs[docs['Claim'] == query].iloc[0].get("Text"):
                        reranked_relevant += 1
                else:
                    for i, row in docs[docs['Claim'] == query].iterrows():
                        if item in row.get("Text"):
                            reranked_relevant += 1

            hybrid_reranked = reranked_relevant / K

            # Add the scores to the lists of all scores
            bm25_all.append(bm25_r_k)
            st_du_all.append(st_du_r_k)
            hybrid_du_all.append(hybrid_du_r_k)
            hybrid_du_reranked.append(hybrid_reranked)

        print(f'average P@{K} for BM25 = {sum(bm25_all) / len(bm25_all)}')
        print(f'average P@{K} for ST-DU = {sum(st_du_all) / len(st_du_all)}')
        print(f'average P@{K} for Hybrid-DU = {sum(hybrid_du_all) / len(hybrid_du_all)}')
        print(f'average P@{K} for Hybrid-DU-reranked = {sum(hybrid_du_reranked) / len(hybrid_du_reranked)}')

        with open('reranking_eval.txt', 'a') as f:
            f.write(f'average P@{K} for Hybrid-DU = {sum(hybrid_du_all) / len(hybrid_du_all)}\naverage P@{K} for Hybrid-DU-reranked = {sum(hybrid_du_reranked) / len(hybrid_du_reranked)}\n')

main()
