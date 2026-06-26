from fastapi import APIRouter

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)


@router.get("/health")
def datasets_health_check():
    return {"message": "Datasets router is working"}