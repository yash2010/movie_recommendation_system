import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

@dataclass
class Moviematch:
    rank: int
    title: str
    year: int
    genre: str
    director: str
    plot_summary: str
    score: float

    def display(self):
        print(f"\n# {self.rank}: {self.title} ({self.year})")
        print(f"Genre: {self.genre}")
        print(f"Director: {self.director}")
        print(f"Plot Summary: {self.plot_summary}")
        print(f"Score: {self.score:.4f}")

class Recommender:

    def __init__(self, artifacts_dir:str = "artifacts"):
        artifacts_dir = Path(artifacts_dir)

        print("Loading artifacts...")
        self.movies = pd.read_parquet(artifacts_dir / "movies.parquet")
        self.embeddings = np.load(artifacts_dir / "embeddings.npy")
        model_name = (artifacts_dir / "model_name.txt").read_text().strip()
        self.model = SentenceTransformer(model_name)
        print(f"Ready. {len(self.movies)} movies loaded.")

    def search(self, query:str, top_k: int = 10, genre_filter: str = None) -> list[Moviematch]:
        query_embed = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = (self.embeddings @ query_embed).copy() 
    
        if genre_filter:
            genre_mask = self.movies["genre"].str.contains(genre_filter, case=False, na=False).to_numpy()
            scores = scores * genre_mask

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            movie = self.movies.iloc[idx]
            results.append(Moviematch(
                rank=rank,
                title=movie["title"],
                year=int(movie["release_year"]),
                genre=movie["genre"],
                director=movie["director"],
                plot_summary=movie["plotsummary"],
                score=float(scores[idx])
            ))
        return results
    
    def similar_to(self, movie_id: int, top_k: int = 5) -> list[Moviematch]:
        idx_matches = self.movies.index[self.movies["movie_id"] == movie_id]
        if len(idx_matches) == 0:
            raise KeyError(f"movie_id {movie_id} not found")

        i = idx_matches[0]

        # Make a COPY — never modify the original scores array
        scores = (self.embeddings @ self.embeddings[i]).copy()
        scores[i] = -np.inf

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            movie = self.movies.iloc[idx]
            results.append(Moviematch(
                rank = rank,
                title = movie["title"],
                year = int(movie["release_year"]),
                genre = movie["genre"],
                director = movie["director"],
                plot_summary = movie["plotsummary"],
                score = float(scores[idx]),
            ))
        return results
if __name__ == "__main__":
    rec = Recommender()
    test_queries = [
        "a dark psychological thriller where you can't trust what's real",
        "a feel-good comedy about friendship and road trips",
        "a science fiction film about artificial intelligence and humanity",
        "Robert De Niro playing a violent gangster",
    ]
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        results = rec.search(query, top_k=5)
        for r in results:
            r.display()

