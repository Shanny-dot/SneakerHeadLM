import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForQuestionAnswering

# -------------------------------
# STEP 1: Load Your Sneaker Data
# -------------------------------
print("📥 Loading sneaker data from CSV...")
df = pd.read_csv("sneaker_knowledge_base.csv")

# Combine important fields into one document per sneaker
def row_to_text(row):
    return (
        f"Name: {row['name']} | Brand: {row['brand']} | Colorway: {row['colorway']} | "
        f"Release Date: {row['releaseDate']} | Retail Price: ${row['retailPrice']} | "
        f"Style ID: {row['styleID']} | Story: {row['description']}"
    )

docs = df.apply(row_to_text, axis=1).tolist()

# -------------------------------
# STEP 2: Create Embeddings
# -------------------------------
print("🔢 Generating embeddings using SentenceTransformers...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, show_progress_bar=True)

# -------------------------------
# STEP 3: Create FAISS Index
# -------------------------------
dimension = doc_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
print(f"✅ FAISS index created with {index.ntotal} sneaker entries!")

# -------------------------------
# STEP 4: Load Fine-Tuned QA Model
# -------------------------------
print("🧠 Loading SneakerHeadLM QA Model...")
model_path = "qa_model/sneaker_qa_model"
qa_model = DistilBertForQuestionAnswering.from_pretrained(model_path)
qa_tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

# -------------------------------
# STEP 5: Retrieval-Augmented QA
# -------------------------------
def retrieve_context(query, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [docs[i] for i in indices[0]]

    # Optional keyword boost
    if "travis" in query.lower():
        boosted = [doc for doc in retrieved if "travis" in doc.lower()]
        boosted += [doc for doc in retrieved if doc not in boosted]
        retrieved = boosted[:top_k]

    return retrieved

def rag_answer(query):
    # Step 1: Retrieve contexts using semantic search
    retrieved_docs = retrieve_context(query)

    # Step 2: Score similarity manually using Sentence Transformers
    best_doc = ""
    best_score = -1

    for doc in retrieved_docs:
        result = qa_pipeline(question=query, context=doc)
        score = result["score"]
        if score > best_score:
            best_score = score
            best_doc = doc

    # Step 3: ONLY use the best_doc now
    final_result = qa_pipeline(question=query, context=best_doc)
    answer = final_result["answer"]

    # Step 4: Add a price fallback if needed
    if "price" in query.lower() and not re.search(r"\$\d+", answer):
        price_match = re.search(r"Retail Price: \$\d+", best_doc)
        if price_match:
            return price_match.group()

    print(f"\n🧾 Best-scoring context:\n{best_doc}\n")
    return answer

# -------------------------------
# STEP 6: Interactive Q&A Loop
# -------------------------------
while True:
    print("\n💬 Ask a sneaker question (or type 'exit'):")
    user_q = input("👟 > ")
    if user_q.lower() == "exit":
        break
    answer = rag_answer(user_q)
    print(f"⚡ SneakerHeadLM says: {answer}")