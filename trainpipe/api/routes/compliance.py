from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...compliance.forget import scan_datasets_for_term
from ...core.db import Database
from ..auth import require_api_key
from ..deps import get_db

router = APIRouter(
    prefix="/compliance",
    tags=["compliance"],
    dependencies=[Depends(require_api_key)],
)


class ForgetScanRequest(BaseModel):
    term: str = Field(min_length=1)
    is_regex: bool = False
    case_sensitive: bool = False


@router.post("/forget-scan")
async def forget_scan(
    body: ForgetScanRequest,
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    async with db.connect() as conn:
        try:
            report = await scan_datasets_for_term(
                conn,
                body.term,
                is_regex=body.is_regex,
                case_sensitive=body.case_sensitive,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
    return report.to_dict()
