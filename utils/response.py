from typing import Any, Optional, Union

class AppResponse:
    def __init__(
        self,
        success: bool = False,
        message: Optional[str] = None,
        data: Optional[Any] = None,
        error: Optional[Exception] = None,
        server_error: Optional[bool] = None,
        http_status_code: Optional[int] = 200
    ) -> None:
        self.success = success
        self.message = message
        self.data = data
        self.error = error
        self.server_error = server_error
        self.http_status_code = http_status_code

class SuccessResponse(AppResponse):
    def __init__(self, message: Union[str, None] = None, data: Union[Any, None] = None, http_status_code: Optional[int] = 200) -> None:
        super().__init__(success=True, message=message, data=data, http_status_code=http_status_code)

class ErrorResponse(AppResponse):
    def __init__(
        self,
        message: Union[str, None] = None,
        error: Union[Exception, None] = None,
        data: Union[Any, None] = None,
        http_status_code: Optional[int] = 400
    ) -> None:
        super().__init__(success=False, message=message, error=error, server_error=False, data=data, http_status_code=http_status_code)

class ServerErrorResponse(AppResponse):
    def __init__(
        self,
        message: Union[str, None] = None,
        error: Union[Exception, None] = None,
        data: Union[Any, None] = None,
        http_status_code: Optional[int] = 500
    ) -> None:
        super().__init__(success=False, message=message, error=error, server_error=True, data=data, http_status_code=http_status_code)
