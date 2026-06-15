import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from recommender import Recommender
from expanders.base import BaseExpander
from expanders.ollama_expander import OllamaExpander
from expanders.local_expander import LocalExpander

recommender: Recommender = None
expander: BaseExpander = None

def _load_expander() -> BaseExpander:

    # set EXPANDER = local to use the model built and trained from scratch
    # the default expander is Ollama 

    mode = os.environ.get("EXPANDER", "ollama").lower()
    if mode == "local":
        try:
            exp = LocalExpander()
            print("Using local expander")
            return exp
        except FileNotFoundError:
            print("Local model not found... falling back to ollama")
    
    try:
        exp = OllamaExpander()
        print("Using OllamaExpander")
        return exp
    except Exception as e:
        print(f"Ollama not available ({e}) — no expansion")
        return None
    
# Runs at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender, expander

    print("Loading reommender...")
    recommender = Recommender()

    print("Loading expander...")
    expander = _load_expander()

    print("Ready")
    yield


    
app = FastAPI(title="Semantic Movie Recommender", description="Find movies by describing what you want.", version="1.0.0", lifespan=lifespan,)

# Response models
class RecommendRequest(BaseModel):
    query:str = Field(..., min_length=3, description="Free-text movie description")
    top_k:int = Field(5, ge=1, le=20)
    genre_filter:str|None = Field(None, description="Filter by genre e.g. 'drama'")
    expand_query:bool = Field(True,  description="Use LLM to expand vague queries")

class MovieResult(BaseModel):
    rank: int
    title: str
    year: int
    genre: str
    director: str
    plot_summary: str
    score: float

class RecommendResponse(BaseModel):
    query: str
    expanded_query: str|None
    results: list[MovieResult]
    took_ms: float
    

class SimilarResponse(BaseModel):
    movie_id: int
    title: str
    results: list[MovieResult]


# End points
@app.get("/health")
def health():
    return{"status": "ok",
           "movie_indexed": len(recommender.movies) if recommender else 0,
           "expander": type(expander).__name__ if expander else "None",}

@app.post("/recommend", response_model = RecommendResponse)
def recommend(req: RecommendRequest):
    start = time.time()

    # optionally expand the query
    expanded_query = None
    search_query = req.query

    if req.expand_query and expander is not None:
        expanded = expander.expand(req.query)
        if expanded != req.query:
            expanded_query = expanded
            search_query = expanded
    
    print(f"DEBUG: searching for: {search_query[:50]}")
    
    
    # search
    results =recommender.search(query=search_query, top_k=req.top_k, genre_filter=req.genre_filter)
    print(f"DEBUG: got {len(results)} results, first score: {results[0].score if results else 'NO RESULTS'}")
    print(f"DEBUG: first title: {results[0].title if results else 'NO RESULTS'}")

    took_ms = (time.time() - start) * 1000
    return RecommendResponse(query=req.query, expanded_query=expanded_query,
                             results=[MovieResult(
                                    rank = r.rank,
                                    title= r.title,
                                    year= r.year,
                                    genre= r.genre,
                                    director= r.director,
                                    plot_summary= r.plot_summary[:300],
                                    score= r.score,)
                                 
                                for r in results                    
                             ],
                             took_ms= round(took_ms, 2),
                             )

@app.get("/similar/{movie_id}", response_model = SimilarResponse)
def similar(movie_id: int, top_k: int = Query(default=5, ge=1, le=20)):
    try:
        results = recommender.similar_to(movie_id, top_k=top_k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")

    # Get the title of the source movie
    movie_row = recommender.movies[recommender.movies["movie_id"] == movie_id]
    title     = movie_row.iloc[0]["title"] if len(movie_row) > 0 else "Unknown"

    return SimilarResponse(movie_id = movie_id, title = title,
                           results=[MovieResult(
                                    rank = r.rank,
                                    title= r.title,
                                    year= r.year,
                                    genre= r.genre,
                                    director= r.director,
                                    plot_summary= r.plot_summary[:300],
                                    score= r.score,)
                                for r in results                    
                             ],)


@app.get("/movies/search")
def search_movies(
    title: str = Query(..., min_length=2, description="Movie title to search"),
    limit: int = Query(default=10, ge=1, le=50),
):
    # search movies by title
    mask    = recommender.movies["title"].str.contains(title, case=False, na=False)
    matches = recommender.movies[mask].head(limit)

    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"No movies found matching '{title}'")

    return {
        "query":   title,
        "results": [
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "year": int(row["release_year"]),
                "genre": row["genre"],
                "director": row["director"],
            }
            for _, row in matches.iterrows()
        ]
    }
