import os
from dotenv import load_dotenv
from google import genai

# --- Setup: connect to Gemini API ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# --- Step 1: Read the HR policy document ---
with open("Company_HR_Policy.txt", "r") as file:
    full_text = file.read()

# --- Step 2: Chunk the text (with overlap) ---
chunk_size = 1000
overlap = 100
step = chunk_size - overlap
chunks = []

for start in range(0, len(full_text), step):
    chunk = full_text[start:start + chunk_size]
    chunks.append(chunk)

# print(f"Total chunks created: {len(chunks)}")

# --- Step 3: Convert each chunk into an embedding ---
embeddings = []
for chunk in chunks:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    embeddings.append(result.embeddings[0].values)

# print(f"Total embeddings created: {len(embeddings)}")
# print(embeddings[0])

import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)


# --- Step 4: Ask a question and find the closest matching chunk ---

def find_best_chunks(question, top_k=3):
    question_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    ).embeddings[0].values

    scores = []
    for i in range(len(chunks)):
        score = cosine_similarity(question_embedding, embeddings[i])
        scores.append((score, chunks[i]))

    scores.sort(reverse=True, key=lambda x: x[0])

    return scores[:top_k]


def generate_answer(question):
    top_chunks = find_best_chunks(question, top_k=3)
    context = "\n\n".join([chunk for score, chunk in top_chunks])

    prompt = f"""Answer the question using only the information in the context below. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

while True:
    question = input("\nAsk a question about the HR policy (or type 'exit' to quit): ")
    
    if question.lower() == "exit":
        print("Goodbye!")
        break
    
    print(generate_answer(question))