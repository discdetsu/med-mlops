import uuid
from datetime import datetime
from typing import Literal

from django.http import HttpRequest
from ninja import Router, Schema, Status
from pydantic import Field

from cxr_backend.feedback.models import Feedback
from cxr_backend.jobs.models import InferenceJob
from cxr_backend.jobs.schemas import Error
from cxr_backend.monitoring.metrics import FEEDBACK

router = Router(tags=["feedback"])


class FeedbackIn(Schema):
    verdict: Literal["agree", "correct"]
    # Required for "correct": every class the model outputs, true when the finding is present.
    labels: dict[str, bool] | None = None
    comment: str = Field("", max_length=2000)


class FeedbackOut(Schema):
    id: int
    job_id: uuid.UUID
    model_version: str
    verdict: str
    labels: dict[str, int]
    comment: str
    created_at: datetime


@router.post(
    "/{uuid:job_id}/feedback", response={201: FeedbackOut, 404: Error, 409: Error, 422: Error}
)
def create_feedback(
    request: HttpRequest, job_id: uuid.UUID, payload: FeedbackIn
) -> Status[FeedbackOut | Error]:
    job = InferenceJob.objects.select_related("prediction").filter(id=job_id).first()
    if job is None:
        return Status(404, Error(detail="job not found"))
    if job.status != InferenceJob.Status.SUCCEEDED:
        return Status(409, Error(detail="feedback needs a finished prediction"))
    prediction = job.prediction

    if payload.verdict == "agree":
        positives = set(prediction.positives)
        labels = {c: int(c in positives) for c in prediction.classes}
    else:
        if payload.labels is None or set(payload.labels) != set(prediction.classes):
            return Status(422, Error(detail=f"labels must give every class: {prediction.classes}"))
        labels = {c: int(payload.labels[c]) for c in prediction.classes}

    feedback = Feedback.objects.create(
        job=job,
        model_version=prediction.model_version,
        verdict=payload.verdict,
        labels=labels,
        comment=payload.comment,
    )
    FEEDBACK.labels(feedback.verdict, feedback.model_version).inc()
    return Status(
        201,
        FeedbackOut(
            id=feedback.id,
            job_id=job.id,
            model_version=feedback.model_version,
            verdict=feedback.verdict,
            labels=feedback.labels,
            comment=feedback.comment,
            created_at=feedback.created_at,
        ),
    )
