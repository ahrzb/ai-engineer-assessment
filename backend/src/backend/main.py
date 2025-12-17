from typing import List, Annotated, Sequence

from functools import lru_cache

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.orm_models import AddressORM
from backend.database import Base, engine, SessionLocal
from backend.models import Address, AddressCreate, AddressesRefresh, AddressUpdate
from backend.mapbox_client import MapboxClient
from backend.similarity import (
    AddressSimilarity,
    build_default_similarity,
)

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
SimilarityEngine = AddressSimilarity


@lru_cache(maxsize=1)
def get_similarity_engine() -> SimilarityEngine:
    # Singleton engine for DI.
    return build_default_similarity()


@lru_cache(maxsize=1)
def get_mapbox_client() -> MapboxClient:
    return MapboxClient(similarity=get_similarity_engine())


MapboxDep = Annotated[MapboxClient, Depends(get_mapbox_client)]
SimilarityEngineDep = Annotated[SimilarityEngine, Depends(get_similarity_engine)]


@app.get("/addresses", response_model=List[Address])
def get_addresses(session: DBSession) -> List[Address]:
    addresses: Sequence[AddressORM] = session.scalars(select(AddressORM)).all()
    return [address.to_pydantic() for address in addresses]


@app.get("/addresses/{id}", response_model=Address)
def get_address(session: DBSession, id: int) -> Address:
    address = session.scalars(select(AddressORM).where(AddressORM.id == id)).one_or_none()
    return address.to_pydantic()


@app.post("/addresses", response_model=Address, status_code=201)
def create_address(
    session: DBSession,
    payload: AddressCreate,
    mapbox: MapboxDep,
    engine: SimilarityEngineDep,
) -> Address:
    match = mapbox.geocode_best_match(payload.address)
    score = engine.score(payload.address, match)
    session.add(address := AddressORM(address=payload.address, matched_address=match, match_score=score))
    session.commit()
    return address.to_pydantic()

@app.post("/addresses/refresh", status_code=200)
def refresh_addresses(
    session: DBSession,
    payload: AddressesRefresh,
    mapbox: MapboxDep,
    engine: SimilarityEngineDep,
):
    query = select(AddressORM)
    if payload.ids is not None:
        query.where(AddressORM.id.in_(payload.ids))
    addresses: Sequence[AddressORM] = session.scalars(query).all()
    for address in addresses:
        match = mapbox.geocode_best_match(address.address)
        score = engine.score(address.address, match)
        address.matched_address = match
        address.match_score = score
    session.commit()
    return

@app.post("/addresses/{id}", response_model=Address, status_code=201)
def update_address(
    session: DBSession,
    id: int,
    payload: AddressUpdate,
    mapbox: MapboxDep,
    engine: SimilarityEngineDep,
) -> Address:
    address = session.scalars(select(AddressORM).where(AddressORM.id == id)).one_or_none()
    match = mapbox.geocode_best_match(payload.address)
    score = engine.score(payload.address, match)
    address.address = payload.address
    address.matched_address = match
    address.match_score = score
    session.commit()
    return address.to_pydantic()

