from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ReviewRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        example=[
            "Absolutely thrilled with this purchase! The build quality exceeds expectations and setup was completely effortless. Battery life easily lasts 2 full days under heavy load. Customer support was extremely helpful and shipping arrived two days early!",
        ],
        description="List of review texts to analyze.",
    )


class ReviewDetail(BaseModel):
    text: str
    prediction: str
    confidence: float
    emotion: str
    aspect_impact: Dict[str, float]
    priority_level: str
    high_risk: bool
    auto_replay: str


class ReviewDashboardResponse(BaseModel):
    summary: Dict[str, Any]
    detailed_results: List[ReviewDetail]
