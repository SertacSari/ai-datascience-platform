from pydantic import BaseModel


class AuthMessageResponse(BaseModel):
    message: str
