import bcrypt
import jwt 
from datetime import datetime, timedelta

SECRET_KEY = "e2c7f3f7d9b8c2e6f0a4d8b2f3a1c9e7d6b2c5a8d1e0f9b3c4e7d1b6c2f8a9d0"  # TODO: Change in production
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=1))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
