from fastapi import APIRouter

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
)


@router.get("/health")
def analysis_health_check():
    return {"message": "Analysis router is working"}