from fastapi import FastAPI
from app import models
from app.routers import analysis, auth, datasets, reports

app = FastAPI(
    title="CS395 Data Science Backend",
    description="Backend API for CSV upload, analysis, ML results, and AI explanations.",
    version="0.1.0",
)

app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(analysis.router)
app.include_router(reports.router)


@app.get("/")
def root():
    return {"message": "CS395 backend is running"}