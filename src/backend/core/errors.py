class ServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class UpstreamError(ServiceError):
    def __init__(self, message: str):
        super().__init__("UPSTREAM_ERROR", message, status_code=502)

class LLMParserError(ServiceError):
    def __init__(self, message: str):
        super().__init__("LLM_PARSER_ERROR", message, status_code=500)

class FrameIdError(ServiceError):
    def __init__(self, message: str):
        super().__init__("INVALID_FRAME_ID", message, status_code=400)

class UnsupportedImageTypeError(ServiceError):
    def __init__(self, message: str):
        super().__init__("UNSUPPORTED_IMAGE_TYPE", message, status_code=415)

class ImageTooLargeError(ServiceError):
    def __init__(self, message: str):
        super().__init__("IMAGE_TOO_LARGE", message, status_code=413)

class ImageValidationError(ServiceError):
    def __init__(self, message: str):
        super().__init__("IMAGE_VALIDATION_ERROR", message, status_code=400)
