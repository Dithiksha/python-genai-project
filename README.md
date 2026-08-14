# Document RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions about any text document by retrieving the most relevant sections and generating accurate, context-grounded responses — with embeddings, chunking, and similarity search implemented manually using NumPy, without relying on a vector database library.

## What It Does

This project lets you have a real conversation with a document instead of manually searching through it. Point it at any plain-text document, ask a question in plain English, and the chatbot finds the relevant sections and generates a clear, accurate answer — grounded strictly in that document's actual content. The included sample document is a fictional company's HR policy manual, used purely to demonstrate the pipeline.

## How It Works

1. **Reading** — The source document is loaded as plain text.
2. **Chunking** — The text is split into overlapping chunks (1000 characters each, 100-character overlap) so that no sentence or fact gets cut off and loses meaning at a chunk boundary.
3. **Embedding** — Each chunk is converted into a 3072-dimensional vector using Gemini's embedding model (`gemini-embedding-001`), capturing its semantic meaning as numbers.
4. **Retrieval** — When a question is asked, it's embedded the same way, then compared against every chunk's embedding using **manually implemented cosine similarity** (built with NumPy — no external vector database). The **top 3** most relevant chunks are retrieved, not just the single best match, to reduce the chance of missing relevant information split across chunks.
5. **Augmented Generation** — The retrieved chunks are inserted into a prompt alongside the original question, and sent to Gemini (`gemini-3.6-flash`) to generate a final answer.
6. **Hallucination Guard** — The prompt explicitly instructs the model to answer *only* using the provided context, and to say "I don't know" if the answer isn't present — rather than confidently guessing or making something up.

The result is an interactive terminal chatbot: ask as many questions as you like, and type `exit` to quit.

## Why No Vector Database?

This project originally used ChromaDB, but hit a persistent, unresolved crash (`0xC0000005` access violation) specific to ChromaDB on Windows. Rather than depending on a third-party library with an unreliable install, the retrieval step was rebuilt from scratch using NumPy — which also means every part of the RAG pipeline here is fully transparent and understood line-by-line, not hidden behind a library.

## Tech Stack

- Python
- Google Gemini API (`google-genai`) — for embeddings and generation
- NumPy — for cosine similarity calculations
- python-dotenv — for API key management

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install google-genai python-dotenv numpy
   ```
3. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
4. Run the chatbot:
   ```
   python read_policy.py
   ```
5. Ask questions about the HR policy document, or type `exit` to quit.

## Example

Using the included sample document (a fictional company's HR policy):

```
Ask a question about the document (or type 'exit' to quit): how many leaves do employees get?

Based on the provided context, the number of leaves depends on the type of leave:
- Sick Leave: 12 days per calendar year
- Casual Leave: 8 days per year
- Annual/Earned Leave: 18 days per year
- Maternity Leave: 26 weeks of paid leave
- Paternity Leave: 2 weeks of paid leave
```

## Using Your Own Document

Replace `Company_HR_Policy.txt` with any plain-text `.txt` file, and update the filename in `read_policy.py`'s file-reading line. The rest of the pipeline works unchanged, regardless of the document's content.

## Project Files

- `read_policy.py` — main chatbot logic (chunking, embedding, retrieval, generation)
- `Company_HR_Policy.txt` — sample document used for testing (fictional company, no real data)
- `test_api.py` — standalone script to verify Gemini API connectivity
