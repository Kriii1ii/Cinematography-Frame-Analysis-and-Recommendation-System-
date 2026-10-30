import os
import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "database.json")

def load_users():
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading database.json: {e}")
        return []

def save_users(users):
    try:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing to database.json: {e}")

class RegisterSchema(BaseModel):
    name: str
    email: str
    password: Optional[str] = "Password123"
    role: Optional[str] = ""
    bio: Optional[str] = ""
    photo: Optional[str] = None
    ref: Optional[str] = None

class LoginSchema(BaseModel):
    email: str
    password: str

@router.get("/api/auth/users")
async def get_all_users():
    return load_users()

import urllib.parse

def generate_initials_avatar(name: str) -> str:
    initial = (name.strip()[0] if name and name.strip() else 'D').upper()
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="240" viewBox="0 0 200 240"><rect width="100%" height="100%" fill="#181816"/><circle cx="100" cy="90" r="45" fill="#C83226"/><text x="100" y="105" font-family="sans-serif" font-size="44" font-weight="bold" fill="#FAF7F2" text-anchor="middle">{initial}</text><rect x="40" y="155" width="120" height="60" rx="30" fill="#2A2723"/></svg>'
    return 'data:image/svg+xml;utf8,' + urllib.parse.quote(svg)

@router.post("/api/auth/register")
async def register_user(data: RegisterSchema):
    if not data.email or not data.email.strip():
        raise HTTPException(status_code=400, detail="Work email is required.")
    if not data.name or not data.name.strip():
        raise HTTPException(status_code=400, detail="Director alias or name is required.")
    if not data.password or len(data.password.strip()) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters long.")

    users = load_users()
    existing = next((u for u in users if u.get("email", "").lower() == data.email.strip().lower()), None)

    user_obj = {
        "id": existing.get("id") if existing else f"USR-{len(users)+1:04d}",
        "name": data.name.strip(),
        "email": data.email.strip(),
        "role": data.role if data.role is not None else "",
        "bio": data.bio if data.bio is not None else "",
        "photo": data.photo or (existing.get("photo") if existing else generate_initials_avatar(data.name)),
        "password": data.password.strip(),
        "ref": data.ref or (existing.get("ref") if existing else f"REF. {data.name[:2].upper()}-8840 • PARIS // BERLIN"),
        "registeredAt": existing.get("registeredAt") if existing else os.environ.get("CURRENT_TIME", "2026-10-06T00:00:00Z")
    }

    if existing:
        idx = users.index(existing)
        users[idx] = user_obj
    else:
        users.append(user_obj)

    save_users(users)
    return {"status": "success", "user": user_obj}

@router.post("/api/auth/login")
async def login_user(data: LoginSchema):
    email = data.email.strip() if data.email else ""
    password = data.password if data.password else ""

    if not email:
        raise HTTPException(status_code=400, detail="Email address is required.")
    if not password:
        raise HTTPException(status_code=400, detail="Password is required.")

    users = load_users()
    existing = next((u for u in users if u.get("email", "").lower() == email.lower()), None)

    if not existing:
        raise HTTPException(status_code=404, detail="No membership found for this email. Please register first.")

    stored_pass = existing.get("password", "")
    if stored_pass and stored_pass != password:
        raise HTTPException(status_code=401, detail="Invalid password. Please check your credentials.")

    return {"status": "success", "user": existing}
