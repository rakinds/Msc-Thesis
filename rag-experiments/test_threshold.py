import os
import bm25s
import pandas as pd
import torch
from transformers import pipeline,  BertLMHeadModel, BertTokenizerFast
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification
    

def split_paragraphs(docs):
    """Split each document in the dataset into paragraphs and 
    turn into a single list of documents (corpus)"""
    corpus = []

    for index, row in docs.iterrows():
        text = row['Text'].replace('\r', '\n')
        text = text.replace('\n\n\n\n', '\n')
        text = text.replace('\n\n\n', '\n')
        text = text.replace('\n ', '\n')
        text_list = text.split('\n')
        for item in text_list:
            if item.count('.') < 2 and item != '':
                item = corpus[-1] + ' ' + item
                corpus[-1] = item
            elif item != '':
                corpus.append(item)
    return corpus


def retrieve_documents(corpus, query):
    """Given a number of docs to retrieve, a corpus and a query, retrieve top documents"""
    # BM25 retriever
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))
    bm25_results, bm25_scores = retriever.retrieve(bm25s.tokenize(query), k=100)

    token = '' # add HuggingFace token

    # ST_DU model
    du_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', token=token)
    du_passages = du_model.encode(corpus, show_progress_bar=True)
    query_embedding = du_model.encode(query)
    similarity = du_model.similarity(query_embedding, du_passages)

    # Zip similarities and corresponding documents
    du_zipped = list(zip(corpus, list(similarity[0])))
    du_zipped = sorted(du_zipped, key=lambda x: x[1], reverse=True)

    # Zip hybrids
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
    
    hybrid_du_sorted = sorted(hybrid_du, key=lambda tup: tup[1], reverse=True)[:100]
    results = [item[0] for item in hybrid_du_sorted]
    scores  = [item[1] for item in hybrid_du_sorted]
    return results, scores


def rerank_results(K, query, documents):
    """Rerank the K retrieved documents using jina-reranker-v2-base-multilingual"""
    model = AutoModelForSequenceClassification.from_pretrained(
        'jinaai/jina-reranker-v2-base-multilingual',
        torch_dtype="auto",
        trust_remote_code=True,
    )  
    model.to('cuda')
    model.eval()

    # construct sentence pairs and compute scores
    sentence_pairs = [[query, doc] for doc in documents]
    scores = model.compute_score(sentence_pairs, max_length=1024)

    # reorder scored documents
    zipped_scores = zip(documents, scores)
    reordered_docs_scores = sorted(zipped_scores, key=lambda tup: tup[1], reverse=True)[:K]
    reordered_docs = [item[0] for item in reordered_docs_scores]
    reordered_scores = [item[1] for item in reordered_docs_scores]

    return reordered_docs, reordered_scores


def main():
    # Read in data and create corpus of concatenated text and claims
    docs = pd.read_csv('') # add data file
    corpus = split_paragraphs(docs)
    K = 3
    
    ood_queries = [
        "Vaccins veroorzaken autisme.",
        "Je kunt kanker genezen door alleen rauwe groenten te eten.",
        "De Holocaust heeft nooit plaatsgevonden.",
        "5G-netwerken verspreiden COVID-19.",
        "Je telefoon luistert altijd naar je, zelfs als hij uitstaat.",
        "Alle politici zijn corrupt.",
        "Geld bijdrukken maakt iedereen rijker.",
        "De beurs is gewoon gokken.",
        "Napoleon was extreem klein – daarom had hij een minderwaardigheidscomplex.",
        "Suiker is net zo verslavend als cocaïne."
    ]

    id_queries = [
        "De polen smelten helemaal niet, er komt juist ijs bij.",
        "Het is niet erg dat er meer CO-2 is, want dat is juist goed voor de planten.",
        "Er is geen versnelde zeespiegelstijging.",
        "IJsberen op de Noordpool zijn niet uitgestorven. Integendeel, hun populatie is toegenomen.",
        "Menselijke uitstoot van CO2 is ongeveer 4% van de natuurlijke uitstoot.",
        "Verbranden van biomassa levert een hogere uitstoot op dan kolen, olie en gas.",
        "Al 100 wetenschappelijke publicaties die stellen dat CO2 weinig tot niets vandoen heeft met klimaatverandering",
        "De WEF-sekte heeft verklaard dat de mensen geen recht hebben op een eigen auto en in plaats daarvan kunnen 'lopen of delen'",
        "Begin juli 2014 werd er een record hoeveelheid zeeijs op Antarctica gemeten",
        "Over windmolens: gezondheidsschade, aantasting natuur, landschap, biodiversiteit, giftige stoffen."
    ]
  
    # out of domain
    ood_scores = []
    for ood_query in ood_queries:
        ood_results, ood_old_scores = retrieve_documents(corpus, ood_query)

        # Reorder retrieved top 100 docs
        ood_reranked_results, ood_reranked_scores = rerank_results(K, ood_query, ood_results)
        ood_scores.append(ood_reranked_scores[0])
     
    id_scores = []
    for id_query in id_queries:
        id_results, id_old_scores = retrieve_documents(corpus, id_query)

        # Reorder retrieved top 100 docs
        id_reranked_results, id_reranked_scores = rerank_results(K, id_query, id_results)
        id_scores.append(id_reranked_scores[0])

    ood_sorted = sorted(ood_scores)
    id_sorted = sorted(id_scores)

    result = f'out of domain: {ood_sorted}\nin domain {id_sorted}'

    with open('threshold.txt', 'w') as f:
            f.write(result)

if __name__ == "__main__":
    main()

 
