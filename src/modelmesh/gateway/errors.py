"""OpenAI-compatible JSON error responses for the gateway.

Every failure uses the shape ``{"error": {"message", "type", "code"}}`` so
existing OpenAI clients can display it without special handling.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse


def openai_error(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: str = "invalid_request",
) -> JSONResponse:
    """Build an OpenAI-shaped error response.

    Args:
        status_code: HTTP status code.
        message: Human-readable, secret-free description.
        error_type: OpenAI error type string.
        code: Short machine-readable error code.

    Returns:
        A JSON response with the standard error envelope.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


def unauthorized(message: str = "Invalid or missing API key.") -> JSONResponse:
    """Return a 401 for a missing, invalid or revoked API key."""
    return openai_error(401, message, "invalid_request_error", "invalid_api_key")


def bad_request(message: str, code: str = "invalid_request") -> JSONResponse:
    """Return a 400 for an unsupported or malformed request."""
    return openai_error(400, message, "invalid_request_error", code)


def model_not_found(model_id: str) -> JSONResponse:
    """Return a 404 for an unknown pinned model id."""
    return openai_error(
        404,
        f"Unknown model: {model_id}",
        "invalid_request_error",
        "model_not_found",
    )


def no_available_model() -> JSONResponse:
    """Return a 503 when routing cannot find an eligible endpoint."""
    return openai_error(
        503,
        "No configured model can take this request right now.",
        "server_error",
        "no_available_model",
    )


def upstream_error() -> JSONResponse:
    """Return a 502 for a provider failure, without echoing raw details."""
    return openai_error(
        502,
        "The upstream provider could not complete this request.",
        "server_error",
        "upstream_error",
    )


def internal_error() -> JSONResponse:
    """Return a generic 500; details go to the server log, not the client."""
    return openai_error(
        500,
        "Internal server error.",
        "server_error",
        "internal_error",
    )
