from dataclasses import dataclass

from flask import jsonify


@dataclass
class ApiError(Exception):
    message: str
    status_code: int = 400
    code: str = "bad_request"

    def __post_init__(self):
        super().__init__(self.message)


def error_response(error, request_id=None):
    payload = {
        "error": {
            "code": error.code,
            "message": error.message,
            "requestId": request_id,
        }
    }
    return jsonify(payload), error.status_code
