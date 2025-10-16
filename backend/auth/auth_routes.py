from fastapi import APIRouter, HTTPException, Depends
import psycopg2
from backend.auth.models import SignupModel, LoginModel
from backend.auth.utils import hash_password, verify_password, create_access_token
from backend import db  # Reusing your db connector

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup")
def signup(user: SignupModel):
    if not user.email.endswith("@gmail.com"):
        raise HTTPException(status_code=400, detail="Only Gmail addresses are allowed for now.")

    cursor, conn = db.connect_db()

    cursor.execute("SELECT * FROM users WHERE email = %s;", (user.email,))
    existing_user = cursor.fetchone()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")

    password_hashed = hash_password(user.password)
    cursor.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s);",
        (user.username, user.email, password_hashed)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "User registered successfully."}

@router.post("/login")
def login(user: LoginModel):
    cursor, conn = db.connect_db()

    cursor.execute("SELECT * FROM users WHERE email = %s;", (user.email,))
    user_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user_record:
        raise HTTPException(status_code=404, detail="User not found.")

    user_id, username, email, password_hash, *_ = user_record

    if not verify_password(user.password, password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    access_token = create_access_token(data={"sub": email})
    return {"access_token": access_token, "token_type": "bearer"}
