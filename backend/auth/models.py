from pydantic import BaseModel, EmailStr

class SignupModel(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginModel(BaseModel):
    email: EmailStr
    password: str
