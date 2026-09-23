"""Uniform, safe error envelope: {"error": {"code", "message"}, "request_id"}."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError, NotFoundError
from app.core.request_context import request_id_var

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None = None


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message), request_id=request_id_var.get()
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def _app_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    if isinstance(exc, NotFoundError):
        logger.info("not found", extra={"resource": exc.resource, "reference": exc.reference})
    else:
        logger.info("request rejected", extra={"error_code": exc.code})
    return error_response(exc.status_code, exc.code, exc.message)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Report where the problem is, never echo the submitted input back.
    fields = sorted({".".join(str(p) for p in err.get("loc", ())) for err in exc.errors()})
    return error_response(422, "validation_error", f"Invalid request: {', '.join(fields)}.")


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = {404: "route_not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return error_response(exc.status_code, code, message)


async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    # Full traceback stays in server logs; the client gets a generic message only.
    logger.error("unhandled error", exc_info=exc)
    return error_response(500, AppError.code, AppError.message)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled_error)
