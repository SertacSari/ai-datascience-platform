from fastapi import APIRouter

router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


@router.get("/health")
def reports_health_check():
    return {"message": "Reports router is working"}