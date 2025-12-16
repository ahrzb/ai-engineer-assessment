from typing import Callable, List, Annotated, Sequence

from functools import lru_cache

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.orm_models import AddressORM
from backend.database import Base, engine, SessionLocal
from backend.models import Address, AddressCreate, AddressesRefresh, AddressUpdate
from backend.mapbox_client import MapboxClient

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Address Assessment Backend",
    description="Minimal backend for the Root Sustainability AI/ML Engineer assessment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

DBSession = Annotated[Session, Depends(get_db_session)]
SimilarityScorer = Callable[[str, str], float]


@lru_cache(maxsize=1)
def get_mapbox_client() -> MapboxClient:
    return MapboxClient()


@lru_cache(maxsize=1)
def get_similarity_scorer() -> SimilarityScorer:
    """
    Singleton scorer for DI.

    Prefers GLiNER-backed similarity when available; falls back to baseline.
    """
    try:
        from backend.gliner_similarity import AddressSimilarityWithGLiNER

        scorer = AddressSimilarityWithGLiNER()

        def _score(a: str, b: str) -> float:
            try:
                s = float(scorer(a, b))
            except Exception:
                from backend.similarity import baseline_similarity

                s = float(baseline_similarity(a, b))
            return max(0.0, min(1.0, s))

        return _score
    except Exception:
        from backend.similarity import baseline_similarity

        def _score(a: str, b: str) -> float:
            return float(max(0.0, min(1.0, baseline_similarity(a, b))))

        return _score


MapboxDep = Annotated[MapboxClient, Depends(get_mapbox_client)]
ScorerDep = Annotated[SimilarityScorer, Depends(get_similarity_scorer)]


def score(address: str, matched_address: str, scorer: SimilarityScorer) -> float:
    return float(scorer(address, matched_address))


def lookup_and_score(address: str, mapbox: MapboxClient, scorer: SimilarityScorer) -> tuple[str, float]:
    matched_address = mapbox.geocode_best_match(address)
    similarity_score = score(address, matched_address, scorer)
    return matched_address, similarity_score


@app.get("/addresses", response_model=List[Address])
def get_addresses(session: DBSession) -> List[Address]:
    addresses: Sequence[AddressORM] = session.scalars(select(AddressORM)).all()
    return [address.to_pydantic() for address in addresses]


@app.get("/addresses/{id}", response_model=Address)
def get_address(session: DBSession, id: int) -> Address:
    address = session.scalars(select(AddressORM).where(AddressORM.id == id)).one_or_none()
    return address.to_pydantic()


@app.post("/addresses", response_model=Address, status_code=201)
def create_address(session: DBSession, payload: AddressCreate, mapbox: MapboxDep, scorer: ScorerDep) -> Address:
    match, score = lookup_and_score(payload.address, mapbox, scorer)
    session.add(address := AddressORM(address=payload.address, matched_address=match, match_score=score))
    session.commit()
    return address.to_pydantic()

@app.post("/addresses/refresh", status_code=200)
def refresh_addresses(session: DBSession, payload: AddressesRefresh, mapbox: MapboxDep, scorer: ScorerDep):
    query = select(AddressORM)
    if payload.ids is not None:
        query.where(AddressORM.id.in_(payload.ids))
    addresses: Sequence[AddressORM] = session.scalars(query).all()
    for address in addresses:
        match, score = lookup_and_score(address.address, mapbox, scorer)
        address.matched_address = match
        address.match_score = score
    session.commit()
    return

@app.post("/addresses/{id}", response_model=Address, status_code=201)
def update_address(session: DBSession, id: int, payload: AddressUpdate, mapbox: MapboxDep, scorer: ScorerDep) -> Address:
    address = session.scalars(select(AddressORM).where(AddressORM.id == id)).one_or_none()
    match, score = lookup_and_score(payload.address, mapbox, scorer)
    address.address = payload.address
    address.matched_address = match
    address.match_score = score
    session.commit()
    return address.to_pydantic()

