"""
Quantix — Unified Backend API (v3)
Consolidation of Logistics, Supply Chain, and Portfolio Optimization.
"""

from __future__ import annotations
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config.settings import API_HOST, API_PORT
from db.database import create_tables
from api.v1 import logistics, supply_chain, portfolio

load_dotenv()

app = FastAPI(
    title="Quantix Unified Optimization Platform",
    description="A single backend for Logistics, Supply Chain, and Portfolio Management using Quantum-Hybrid AI.",
    version="3.0.0"
)

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await create_tables()


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Route not found",
            "available_domains": ["/api/v1/logistics", "/api/v1/supply-chain", "/api/v1/portfolio"],
            "docs": "/docs",
        },
    )


@app.get("/", response_model=dict)
async def root():
    return {
        "service": "Quantix Unified AI Platform",
        "status": "running",
        "version": "3.0.0",
        "endpoints": {
            "logistics": "/api/v1/logistics",
            "supply_chain": "/api/v1/supply-chain",
            "portfolio": "/api/v1/portfolio",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=dict)
async def health():
    return {"status": "ok"}


# Include domain-specific routers
app.include_router(logistics.router, prefix="/api/v1")
app.include_router(supply_chain.router, prefix="/api/v1")
app.include_router(portfolio.router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)