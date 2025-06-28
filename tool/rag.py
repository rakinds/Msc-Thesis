"""
rag.py
Input: A misinformed claim about climate change
Output: A generated correction about the misinformation
"""

import os
import bm25s
import pandas as pd
import torch
from transformers import pipeline
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


def create_context(results):
    """Turn three retrieved documents into one string separated with newlines"""
    context = ''
    for item in results:
        context += item + '\n\n'
    print([context])
    return context


def retrieve_documents(corpus, query, token):
    """Given a number of docs to retrieve, a corpus and a query, retrieve top documents"""
    # BM25 retriever
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))
    bm25_results, bm25_scores = retriever.retrieve(bm25s.tokenize(query), k=100)

    # ST_DU model
    du_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
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



def create_prompt(context, query, few_shot):
    """Create a text prompt, include few-shot information if specified"""
    if few_shot:
        prompt = ('Leg uit in een lopende tekst waarom de claim ": '
                + query + '" fout is.\n Wees gedetailleerd, geef '
                'alternatieve uitleg waar mogelijk en eindig met '
                'een herhaling van de misinformatie en de correctie. '
                'Hieronder vind je een kort voorbeeld van een correctie voor de claim "Er is juist een record hoeveelheid ijs op Antarctica".\n\n'
                '"De claim dat er een recordhoeveelheid ijs op Antarctica is, '
                'is incorrect. Hoewel er sprake is van een toename van ijsmassa'
                ' op bepaalde drijvende ijsplaten tussen 2009 en 2019, is dit slechts'
                'een deel van het beeld. De totale ijskap op Antarctica verliest in '
                'feite massa, wat aantoont dat klimaatopwarming een realiteit blijft.'
                ' De toename van ijsmassa op drijvende ijsplaten kan worden verklaard '
                'door seizoensgebonden variaties en natuurlijke cycli. Deze toename zegt'
                ' echter niets over het algemene trend van ijsverlies op het vasteland van '
                'Antarctica. De bewering dat de hoeveelheid ijs op Antarctica toeneemt en '
                'klimaatopwarming daarom onzin is, is misleidend: ondanks seizoensgebonden variaties '
                'neemt de hoeveelheid ijs in zijn geheel af."\n\n'
                'Maak als het nodig is gebruik van onderstaande informatie:\n\n' 
                + context)
    else:
        prompt = ('Leg uit in een lopende tekst waarom de claim ": '
                + query + '" fout is.\n Wees gedetailleerd, geef '
                'alternatieve uitleg waar mogelijk en eindig met '
                'een herhaling van de misinformatie en de correctie. '
                'Maak als het nodig is gebruik van onderstaande informatie:\n\n' 
                + context)
    return prompt


def main():
    # Read in data and create corpus of concatenated text and claims
    docs = pd.read_csv('') # add data file
    corpus = split_paragraphs(docs)
    K = 3
    token = "" # add HuggingFace token
    
    # Query the corpus and get top 100 results
    query = "Het klimaat verandert altijd, dus de huidige opwarming is niet bijzonder."
    results, scores = retrieve_documents(corpus, query, token)

    # Reorder retrieved top 100 docs
    reranked_results, reranked_scores = rerank_results(K, query, results)
    if reranked_scores[0] < 0.3:
        print("I'm sorry, I do not have enough information available to check your claim. Please enter another claim instead.")
        exit()

    # Create context string
    context = create_context(reranked_results)

    # Define model
    model_id = "Qwen/Qwen2.5-32B-Instruct"

    # Define prompt
    few_shot = False
    prompt = create_prompt(context, query, few_shot)

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",  
        token=token
    )

    messages = [
        {"role": "system", "content": "Je bent een behulpzame assistent die misinformatie over het klimaat overtuigend corrigeert op basis van gevonden informatie. Als je over bepaalde informatie niet zeker bent, zeg je dat je de gevraagde informatie niet hebt. Verzin zelf niets. Maak nooit verwijzingen naar figuren, bronnen of andere pagina's."},
        {"role": "user", "content": prompt},
    ]

    outputs = pipe( 
        messages,
        max_new_tokens=256*8,
    )

    print(outputs[0]["generated_text"][-1])
    output_text = f"Model:{model_id}\nRetrieval: Hybrid-DU with reranking\nClaim: {query}\nPrompt: {prompt}\nSystem role: {messages[0]['content']}\n\nOutput:\n{outputs[0]['generated_text'][-1]['content']}"

    with open('correction.txt', 'w') as f:
            f.write(output_text)

if __name__ == "__main__":
    main()

