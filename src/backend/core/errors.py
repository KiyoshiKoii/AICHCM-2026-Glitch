class ServiceError(Exception):
    """Base error that can safely cross the HTTP boundary."""

    status_code = 500
    code = "service_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LLMParserError(ServiceError):
    status_code = 502
    code = "llm_parser_error"


class UpstreamError(ServiceError):
    status_code = 502
    code = "upstream_error"


class FrameIdError(ServiceError):
    status_code = 422
    code = "invalid_frame_id"


class ImageValidationError(ServiceError):
    status_code = 422
    code = "invalid_image"


class UnsupportedImageTypeError(ServiceError):
    status_code = 415
    code = "unsupported_image_type"


class ImageTooLargeError(ServiceError):
    status_code = 413
    code = "image_too_large"
