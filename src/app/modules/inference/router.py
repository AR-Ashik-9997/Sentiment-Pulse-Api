from fastapi import APIRouter, FastAPI, HTTPException
from app.modules.inference.schemas import ReviewRequest, ReviewDashboardResponse
from app.modules.inference.service import run_Sentiment_analysis_inference

router = APIRouter(prefix="/inference", tags=["Sentiment Analysis & Review Dashboard"])


@router.post("/review-dashboard", response_model=ReviewDashboardResponse)
def review_dashboard_endpoint(payload: ReviewRequest):
    try:
        response = run_Sentiment_analysis_inference(payload)
        return response
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred in router: {str(e)}"
        )



