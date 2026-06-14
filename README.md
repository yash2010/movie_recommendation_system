# 🎬 Semantic Movie Recommendation System

A production-ready movie recommendation system that finds films based on natural language descriptions rather than keyword matching. Built from scratch using transformer embeddings, semantic similarity search, and an optional LLM query expansion layer — all served through a REST API.

---

## ✨ Features

- **Semantic search** — finds movies by meaning, not just keywords
- **Query expansion** — automatically enriches vague queries using a local LLM (Ollama)
- **Custom transformer** — seq2seq query expander built from scratch in PyTorch
- **34,608 movies** indexed from Wikipedia
- **REST API** — FastAPI with interactive Swagger docs
- **Two expander modes** — Ollama (high quality) or local trained model (offline)
- **Similar movies** — find movies similar to any movie by ID
- **Genre filtering** — filter results by genre

---

## 🏗️ Architecture

```
User types a description
        │
        ▼
Query Expansion (Ollama / Custom Transformer)
  "a dark thriller" → "A psychologically intense thriller
                        featuring an unreliable narrator..."
        │
        ▼
Sentence Transformer Embedding (all-MiniLM-L6-v2)
  Text → 384-dimensional vector
        │
        ▼
Cosine Similarity Search
  Query vector · Movie vectors (34,608 × 384 matrix)
        │
        ▼
Top-k Results via REST API
```

---

## 📁 Project Structure

```
movie-recommendation/
├── data/
│   ├── movies_clean.csv          # Cleaned movie dataset
│   ├── vague_queries.json        # Queries for training data generation
│   └── training_pairs.json       # Generated (query, expansion) pairs
├── artifacts/
│   ├── embeddings.npy            # (34608, 384) embedding matrix
│   ├── movies.parquet            # Cleaned movie metadata
│   ├── model_name.txt            # Embedding model identifier
│   └── expander/                 # Trained query expander weights
├── model/
│   ├── config.py                 # All hyperparameters
│   ├── tokenizer.py              # Text ↔ token ID conversion
│   ├── dataset.py                # PyTorch Dataset and DataLoader
│   ├── attention.py              # Multi-head attention from scratch
│   ├── blocks.py                 # Encoder and decoder blocks
│   ├── encoder.py                # Full transformer encoder
│   ├── decoder.py                # Full transformer decoder
│   └── model.py                  # Complete seq2seq QueryExpander
├── expanders/
│   ├── base.py                   # Abstract BaseExpander interface
│   ├── ollama_expander.py        # Query expansion via Ollama LLM
│   └── local_expander.py         # Query expansion via trained model
├── build_index.py                # Build and cache movie embeddings
├── clean_data.py                 # Data cleaning pipeline
├── recommender.py                # Core semantic search engine
├── generate_training_data.py     # Generate training pairs using Ollama
├── train.py                      # Training loop for query expander
├── api.py                        # FastAPI REST service
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/movie-recommendation.git
cd movie-recommendation

conda create -n movie_re python=3.11
conda activate movie_re

pip install -r requirements.txt
```

### 2. Get the dataset

Download the Wikipedia Movie Plots dataset and place it in `data/`:

```
data/movies_clean.csv
```

The CSV should have these columns:
`release_year, title, origin_ethnicity, director, cast, genre, wiki_page, plot, plotsummary`

### 3. Clean the data

```bash
python clean_data.py
```

### 4. Build the embedding index

This embeds all 34,608 movies using sentence-transformers. Takes ~8 minutes, runs once.

```bash
python build_index.py
```

### 5. Start Ollama (for query expansion)

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama serve
```

### 6. Start the API

```bash
uvicorn api:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## 📡 API Endpoints

### `POST /recommend`

Find movies matching a natural language description.

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a dark psychological thriller",
    "top_k": 5,
    "genre_filter": "thriller",
    "expand_query": true
  }'
```

**Response:**
```json
{
  "query": "a dark psychological thriller",
  "expanded_query": "A psychologically intense thriller featuring...",
  "results": [
    {
      "rank": 1,
      "title": "After the Dark",
      "year": 2014,
      "genre": "thriller",
      "director": "John Huddles",
      "plot_summary": "...",
      "score": 0.5843
    }
  ],
  "took_ms": 68.06
}
```

### `GET /similar/{movie_id}`

Find movies similar to a given movie.

```bash
curl http://localhost:8000/similar/15853?top_k=5
```

### `GET /movies/search`

Search movies by title to find `movie_id`.

```bash
curl "http://localhost:8000/movies/search?title=inception"
```

### `GET /health`

Check system status.

```bash
curl http://localhost:8000/health
```

---

## 🔧 Configuration

### Switch between expanders

```bash
# Use Ollama (default, best quality)
uvicorn api:app --reload

# Use local trained model (offline, faster)
set EXPANDER=local  # Windows
export EXPANDER=local  # Mac/Linux
uvicorn api:app --reload
```

### Train the custom query expander

```bash
# Generate training data (requires Ollama)
python generate_training_data.py

# Train the model
python train.py
```

---

## 🧠 Custom Transformer

The query expander is a seq2seq transformer built entirely from scratch in PyTorch — no pretrained weights, no Hugging Face models.

| Hyperparameter | Value |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Vocabulary size | 8,000 tokens |
| Embedding dimension | 256 |
| Attention heads | 8 |
| Encoder/Decoder layers | 4 each |
| Feedforward dimension | 512 |
| Total parameters | 11,404,288 |
| Optimizer | AdamW |
| Loss function | CrossEntropyLoss + label smoothing |

---

## 📊 How Semantic Search Works

Traditional keyword search matches exact words. Semantic search matches meaning:

| Query | Keyword Search | Semantic Search |
|---|---|---|
| "dark thriller" | Movies containing "dark" and "thriller" | Movies with themes of dread, tension, moral ambiguity |
| "film that makes you think" | No matches (too vague) | Philosophical dramas, thought-provoking sci-fi |
| "something like Inception" | No matches | Mind-bending sci-fi, non-linear narratives |

The system encodes both movies and queries as 384-dimensional vectors using `all-MiniLM-L6-v2`. Cosine similarity between vectors measures semantic relatedness. Higher score = more similar meaning.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Custom transformer | PyTorch (built from scratch) |
| LLM query expansion | Ollama (llama3) |
| Similarity search | NumPy (cosine similarity) |
| Data processing | Pandas, PyArrow |
| REST API | FastAPI + Uvicorn |
| Data validation | Pydantic |

---

## 🗺️ Roadmap

- [ ] Generate 500+ training pairs and retrain query expander
- [ ] Add user feedback endpoint for interaction logging
- [ ] Integrate collaborative filtering (matrix factorization)
- [ ] Add sequential recommendation model (SASRec)
- [ ] FAISS index for larger corpora (100k+ movies)
- [ ] Docker deployment
- [ ] Add LLM re-ranking layer

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [sentence-transformers](https://www.sbert.net/) for the embedding model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Ollama](https://ollama.com/) for local LLM inference
- Wikipedia Movie Plots dataset
