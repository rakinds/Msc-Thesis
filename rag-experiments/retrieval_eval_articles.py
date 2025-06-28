import bm25s
from sentence_transformers import SentenceTransformer
import pandas as pd

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

    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))

    token = '' # add HF token
    du_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', token=token)
    du_passages = du_model.encode(corpus, show_progress_bar=True)

    p_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', token=token)
    p_passages = p_model.encode(corpus, show_progress_bar=True)

    for K in K_s:
        bm25_all = []
        st_du_all = []
        st_p_all = []
        hybrid_du_all = []
        hybrid_p_all = []

        for query in queries:
            #BM25s
            bm25_results, bm25_scores = retriever.retrieve(bm25s.tokenize(query), k=100)

            #ST-DU
            query_embedding = du_model.encode(query)
            similarity_du = du_model.similarity(query_embedding, du_passages)

            # Zip similarities and corresponding documents
            du_zipped = list(zip(corpus, list(similarity_du[0])))
            du_zipped = sorted(du_zipped, key=lambda x: x[1], reverse=True)
            st_du_results = [du_zipped[i][0] for i in range(K)]

            #ST-P
            query_embedding = p_model.encode(query)
            similarity_p = p_model.similarity(query_embedding, p_passages)

            # Zip similarities and corresponding documents
            p_zipped = list(zip(corpus, list(similarity_p[0])))
            p_zipped = sorted(p_zipped, key=lambda x: x[1], reverse=True)
            st_p_results = [p_zipped[i][0] for i in range(K)]

            # zip for hybrids
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

            #HYBRID-P
            hybrid_p = []
            for item in bm25:
                p_dict = dict(p_zipped)
                p_score = float(p_dict[item[0]])
                score = 0.3 * item[1] + p_score
                hybrid_p.append((item[0], score))
            
            hybrid_p_sorted = sorted(hybrid_p, key=lambda tup: tup[1], reverse=True)[:K]

            # Calculate total number of relevant docs
            if docs[docs['Claim'] == query].shape[0] == 1:
                total_relevant = lens[docs[docs['Claim'] == query].index[0]]
            else:
                total_relevant = 0
                for i, row in docs[docs['Claim'] == query].iterrows():
                    total_relevant += lens[int(i)]

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

            # ST distiluse-base-multilingual-cased-v1
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

            # ST paraphrase-multilingual-mpnet-base-v2
            st_p_relevant = 0
            for item in st_p_results:
                if docs[docs['Claim'] == query].shape[0] == 1:
                    if item in docs[docs['Claim'] == query].iloc[0].get("Text"):
                        st_p_relevant += 1
                else:
                    for i, row in docs[docs['Claim'] == query].iterrows():
                        if item in row.get("Text"):
                            st_p_relevant += 1

            st_p_r_k = st_p_relevant / K

            # Hybrid distiluse-base-multilingual-cased-v1
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
            
            # Hybrid paraphrase-multilingual-mpnet-base-v2
            hybrid_p_relevant = 0
            for item in hybrid_p_sorted:
                if docs[docs['Claim'] == query].shape[0] == 1:
                    if item[0] in docs[docs['Claim'] == query].iloc[0].get("Text"):
                        hybrid_p_relevant += 1
                else:
                    for i, row in docs[docs['Claim'] == query].iterrows():
                        if item[0] in row.get("Text"):
                            hybrid_p_relevant += 1

            hybrid_p_r_k = hybrid_p_relevant / K

            # Add the scores to the lists of all scores
            bm25_all.append(bm25_r_k) 
            st_du_all.append(st_du_r_k)
            st_p_all.append(st_p_r_k)
            hybrid_du_all.append(hybrid_du_r_k)
            hybrid_p_all.append(hybrid_p_r_k)

        print(f'average P@{K} for BM25 = {sum(bm25_all) / len(bm25_all)}')
        print(f'average P@{K} for ST-DU = {sum(st_du_all) / len(st_du_all)}')
        print(f'average P@{K} for ST-P = {sum(st_p_all) / len(st_p_all)}')
        print(f'average P@{K} for Hybrid-DU = {sum(hybrid_du_all) / len(hybrid_du_all)}')
        print(f'average P@{K} for Hybrid-P = {sum(hybrid_p_all) / len(hybrid_p_all)}')

        results = f'K = {K}, BM25 = {sum(bm25_all) / len(bm25_all)}, ST-DU = {sum(st_du_all) / len(st_du_all)}, ST-P = {sum(st_p_all) / len(st_p_all)}, Hybrid-DU = {sum(hybrid_du_all) / len(hybrid_du_all)}, Hybrid-P = {sum(hybrid_p_all) / len(hybrid_p_all)}\n\n'

        with open('retrieval_eval_articles.txt', 'a') as f:
            f.write(results)
main()
