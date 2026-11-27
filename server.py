import asyncio
import logging
import os
import re
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, IndexModel
from starlette.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

client = AsyncIOMotorClient(os.environ['MONGO_URL'])
db = client[os.environ['DB_NAME']]
JWT_SECRET = os.environ['JWT_SECRET']
ADMIN_USERNAME = os.environ['ADMIN_USERNAME']
ADMIN_PASSWORD = os.environ['ADMIN_PASSWORD']
TOKEN_DAYS = 60

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

INDEXES: dict[str, list[IndexModel]] = {
    "users": [
        IndexModel([("id", ASCENDING)], name="id", unique=True),
        IndexModel([("username_lower", ASCENDING)], name="username_lower", unique=True),
        IndexModel([("student_id", ASCENDING)], name="student_id"),
        IndexModel([("professor_id", ASCENDING)], name="professor_id"),
    ],
    "students": [
        IndexModel([("id", ASCENDING)], name="id", unique=True),
        IndexModel([("full_name", ASCENDING)], name="full_name"),
        IndexModel([("professor_id", ASCENDING)], name="professor_id"),
    ],
    "attendance": [
        IndexModel([("date", ASCENDING), ("student_id", ASCENDING)], name="date_student", unique=True),
    ],
    # Agenda de treino é por professor/cidade: cada professor tem seus 7 dias.
    "schedule": [IndexModel([("professor_id", ASCENDING), ("weekday", ASCENDING)], name="prof_weekday", unique=True)],
    "messages": [
        IndexModel([("created_at", DESCENDING)], name="created_desc"),
        IndexModel([("professor_id", ASCENDING)], name="professor_id"),
    ],
}

# Índices antigos que precisam ser removidos ao migrar para o modelo multi-cidade.
OBSOLETE_INDEXES: dict[str, list[str]] = {
    # schedule era global (único por weekday); agora é por (professor_id, weekday).
    "schedule": ["weekday"],
}


async def ensure_indexes() -> None:
    for collection, names in OBSOLETE_INDEXES.items():
        for name in names:
            with suppress(Exception):
                await db[collection].drop_index(name)
    for collection, models in INDEXES.items():
        for model in models:
            try:
                await db[collection].create_indexes([model])
            except Exception as exc:
                logger.error("ensure_indexes(%s.%s): %s", collection, model.document["name"], exc)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_pw(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


async def seed_admin() -> None:
    # Sempre sincroniza usuário/senha do admin (MESTRE) com as variáveis de
    # ambiente, para que trocar ADMIN_USERNAME/ADMIN_PASSWORD no host tenha
    # efeito em cada deploy.
    await db.users.update_one(
        {"role": "admin"},
        {
            "$set": {
                "username": ADMIN_USERNAME, "username_lower": ADMIN_USERNAME.lower(),
                "password_hash": hash_pw(ADMIN_PASSWORD), "name": "Mestre Muringá",
            },
            "$setOnInsert": {
                "id": str(uuid.uuid4()), "student_id": None, "professor_id": None,
                "city": "", "created_at": now_iso(),
            },
        },
        upsert=True,
    )


app = FastAPI()


@app.get("/health", include_in_schema=False)
@app.head("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


api = APIRouter(prefix="/api")
bearer = HTTPBearer(auto_error=False)

# ---------- Models ----------
DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
TIME_RE = r"^([01]\d|2[0-3]):[0-5]\d$"


class LoginIn(BaseModel):
    username: str
    password: str


class ProfessorIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    city: str = Field(min_length=2, max_length=120)
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: Optional[str] = Field(default=None, min_length=4, max_length=100)


class StudentIn(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    birth_date: str = ""
    phone: str = ""
    mother_name: str = ""
    mother_phone: str = ""
    father_name: str = ""
    father_phone: str = ""
    address: str = ""


class AccessIn(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: Optional[str] = Field(default=None, min_length=4, max_length=100)


class AttendanceIn(BaseModel):
    date: str = Field(pattern=DATE_RE)
    student_id: str
    status: Literal["present", "absent", "none"]


class ScheduleIn(BaseModel):
    active: bool
    start: str = Field(default="", pattern=r"^(([01]\d|2[0-3]):[0-5]\d)?$")
    end: str = Field(default="", pattern=r"^(([01]\d|2[0-3]):[0-5]\d)?$")


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


# ---------- Auth ----------
def make_token(user_id: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": user_id, "role": role, "iat": now, "exp": now + timedelta(days=TOKEN_DAYS)},
                      JWT_SECRET, algorithm="HS256")


def scope_pid(user: dict) -> Optional[str]:
    """ID do professor que delimita os dados visíveis a este usuário.

    Professor = o próprio id. Aluno = o professor dono do seu cadastro.
    Admin (MESTRE) = None (sem cidade fixa; vê tudo nos relatórios)."""
    if user["role"] == "professor":
        return user["id"]
    if user["role"] == "student":
        return user.get("professor_id")
    return None


def public_user(u: dict) -> dict:
    return {
        "id": u["id"], "username": u["username"], "role": u["role"], "name": u.get("name", ""),
        "student_id": u.get("student_id"), "city": u.get("city", ""),
        "professor_id": u["id"] if u["role"] == "professor" else u.get("professor_id"),
    }


async def current_user(cred: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    err = HTTPException(status_code=401, detail="Sessão expirada. Entre novamente.")
    if not cred:
        raise err
    try:
        payload = jwt.decode(cred.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise err
    user = await db.users.find_one({"id": payload.get("sub")}, {"_id": 0})
    if not user:
        raise err
    return user


async def admin_only(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Apenas o Mestre pode fazer isso.")
    return user


async def professor_only(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "professor":
        raise HTTPException(status_code=403, detail="Apenas o professor pode fazer isso.")
    return user


async def owned_student(student_id: str, user: dict) -> dict:
    """Garante que o aluno pertence ao professor logado."""
    st = await db.students.find_one({"id": student_id}, {"_id": 0})
    if not st or st.get("professor_id") != user["id"]:
        raise HTTPException(status_code=404, detail="Aluno não encontrado.")
    return st


@api.post("/auth/login")
async def login(data: LoginIn):
    user = await db.users.find_one({"username_lower": data.username.strip().lower()}, {"_id": 0})
    if not user or not verify_pw(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Usuário ou senha incorretos.")
    return {"token": make_token(user["id"], user["role"]), "user": public_user(user)}


@api.get("/auth/me")
async def me(user: dict = Depends(current_user)):
    return public_user(user)


# ---------- Professores (somente o MESTRE) ----------
@api.get("/professors")
async def list_professors(_: dict = Depends(admin_only)):
    profs = await db.users.find({"role": "professor"}, {"_id": 0}).sort("name", 1).to_list(1000)
    out = []
    for p in profs:
        count = await db.students.count_documents({"professor_id": p["id"]})
        out.append({"id": p["id"], "name": p.get("name", ""), "city": p.get("city", ""),
                    "username": p["username"], "students": count})
    return out


@api.post("/professors")
async def create_professor(data: ProfessorIn, _: dict = Depends(admin_only)):
    if not data.password:
        raise HTTPException(status_code=422, detail="Informe uma senha.")
    uname = data.username.strip()
    if await db.users.find_one({"username_lower": uname.lower()}, {"_id": 1}):
        raise HTTPException(status_code=409, detail="Este usuário já está em uso.")
    pid = str(uuid.uuid4())
    await db.users.insert_one({
        "id": pid, "username": uname, "username_lower": uname.lower(),
        "password_hash": hash_pw(data.password), "role": "professor",
        "name": data.name.strip(), "city": data.city.strip(),
        "professor_id": pid, "student_id": None, "created_at": now_iso(),
    })
    return {"ok": True, "id": pid, "username": uname}


@api.put("/professors/{pid}")
async def update_professor(pid: str, data: ProfessorIn, _: dict = Depends(admin_only)):
    prof = await db.users.find_one({"id": pid, "role": "professor"}, {"_id": 0})
    if not prof:
        raise HTTPException(status_code=404, detail="Professor não encontrado.")
    uname = data.username.strip()
    taken = await db.users.find_one({"username_lower": uname.lower()}, {"_id": 0})
    if taken and taken["id"] != pid:
        raise HTTPException(status_code=409, detail="Este usuário já está em uso.")
    upd = {"username": uname, "username_lower": uname.lower(),
           "name": data.name.strip(), "city": data.city.strip()}
    if data.password:
        upd["password_hash"] = hash_pw(data.password)
    await db.users.update_one({"id": pid}, {"$set": upd})
    # Mantém a cidade dos alunos e das contas de aluno em sincronia com o professor.
    await db.students.update_many({"professor_id": pid}, {"$set": {"city": data.city.strip()}})
    await db.users.update_many({"professor_id": pid, "role": "student"}, {"$set": {"city": data.city.strip()}})
    return {"ok": True, "username": uname}


@api.delete("/professors/{pid}")
async def delete_professor(pid: str, _: dict = Depends(admin_only)):
    prof = await db.users.find_one({"id": pid, "role": "professor"}, {"_id": 1})
    if not prof:
        raise HTTPException(status_code=404, detail="Professor não encontrado.")
    students = await db.students.find({"professor_id": pid}, {"_id": 0, "id": 1}).to_list(5000)
    sids = [s["id"] for s in students]
    if sids:
        await db.attendance.delete_many({"student_id": {"$in": sids}})
    await db.students.delete_many({"professor_id": pid})
    await db.schedule.delete_many({"professor_id": pid})
    await db.messages.delete_many({"professor_id": pid})
    # Remove as contas de aluno desse professor e, por fim, o próprio professor.
    await db.users.delete_many({"professor_id": pid, "role": "student"})
    await db.users.delete_one({"id": pid})
    return {"ok": True}


# ---------- Students ----------
@api.get("/students")
async def list_students(user: dict = Depends(current_user)):
    role = user["role"]
    if role == "admin":
        docs = await db.students.find({}, {"_id": 0}).sort("full_name", 1).to_list(5000)
        accounts = await db.users.find({"role": "student"}, {"_id": 0, "student_id": 1, "username": 1}).to_list(5000)
        by_student = {a["student_id"]: a["username"] for a in accounts}
        profs = {p["id"]: p for p in await db.users.find(
            {"role": "professor"}, {"_id": 0, "id": 1, "name": 1, "city": 1}).to_list(1000)}
        for d in docs:
            d["username"] = by_student.get(d["id"])
            prof = profs.get(d.get("professor_id"), {})
            d["professor_name"] = prof.get("name", "")
        return docs
    pid = scope_pid(user)
    docs = await db.students.find({"professor_id": pid}, {"_id": 0}).sort("full_name", 1).to_list(5000)
    if role == "student":
        return [{"id": d["id"], "full_name": d["full_name"]} for d in docs]
    accounts = await db.users.find(
        {"role": "student", "professor_id": pid}, {"_id": 0, "student_id": 1, "username": 1}).to_list(5000)
    by_student = {a["student_id"]: a["username"] for a in accounts}
    for d in docs:
        d["username"] = by_student.get(d["id"])
    return docs


def clean_name(name: str) -> str:
    name = name.strip()
    if len(name) < 2:
        raise HTTPException(status_code=422, detail="Informe o nome completo.")
    return name


@api.post("/students")
async def create_student(data: StudentIn, user: dict = Depends(professor_only)):
    doc = {"id": str(uuid.uuid4()), **data.model_dump(),
           "professor_id": user["id"], "city": user.get("city", ""), "created_at": now_iso()}
    doc["full_name"] = clean_name(doc["full_name"])
    await db.students.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.put("/students/{student_id}")
async def update_student(student_id: str, data: StudentIn, user: dict = Depends(professor_only)):
    await owned_student(student_id, user)
    upd = data.model_dump()
    upd["full_name"] = clean_name(upd["full_name"])
    await db.students.update_one({"id": student_id}, {"$set": upd})
    await db.users.update_many({"student_id": student_id}, {"$set": {"name": upd["full_name"]}})
    return await db.students.find_one({"id": student_id}, {"_id": 0})


@api.delete("/students/{student_id}")
async def delete_student(student_id: str, user: dict = Depends(professor_only)):
    await owned_student(student_id, user)
    await db.students.delete_one({"id": student_id})
    await db.users.delete_many({"student_id": student_id})
    await db.attendance.delete_many({"student_id": student_id})
    return {"ok": True}


@api.put("/students/{student_id}/access")
async def set_access(student_id: str, data: AccessIn, user: dict = Depends(professor_only)):
    student = await owned_student(student_id, user)
    uname = data.username.strip()
    taken = await db.users.find_one({"username_lower": uname.lower()}, {"_id": 0})
    if taken and taken.get("student_id") != student_id:
        raise HTTPException(status_code=409, detail="Este usuário já está em uso.")
    existing = await db.users.find_one({"student_id": student_id}, {"_id": 0})
    if existing:
        upd = {"username": uname, "username_lower": uname.lower(), "name": student["full_name"],
               "professor_id": user["id"], "city": user.get("city", "")}
        if data.password:
            upd["password_hash"] = hash_pw(data.password)
        await db.users.update_one({"id": existing["id"]}, {"$set": upd})
    else:
        if not data.password:
            raise HTTPException(status_code=422, detail="Informe uma senha.")
        await db.users.insert_one({
            "id": str(uuid.uuid4()), "username": uname, "username_lower": uname.lower(),
            "password_hash": hash_pw(data.password), "role": "student", "name": student["full_name"],
            "student_id": student_id, "professor_id": user["id"], "city": user.get("city", ""),
            "created_at": now_iso(),
        })
    return {"ok": True, "username": uname}


@api.delete("/students/{student_id}/access")
async def remove_access(student_id: str, user: dict = Depends(professor_only)):
    await owned_student(student_id, user)
    await db.users.delete_many({"student_id": student_id, "role": "student"})
    return {"ok": True}


# ---------- Attendance ----------
async def scope_student_ids(user: dict) -> list[str]:
    pid = scope_pid(user)
    docs = await db.students.find({"professor_id": pid}, {"_id": 0, "id": 1}).to_list(5000)
    return [d["id"] for d in docs]


@api.get("/attendance")
async def get_attendance(date: str = Query(pattern=DATE_RE), user: dict = Depends(current_user)):
    if user["role"] == "admin":
        docs = await db.attendance.find({"date": date}, {"_id": 0}).to_list(5000)
    else:
        sids = await scope_student_ids(user)
        docs = await db.attendance.find(
            {"date": date, "student_id": {"$in": sids}}, {"_id": 0}).to_list(5000)
    return {"date": date, "records": {d["student_id"]: d["status"] for d in docs}}


@api.put("/attendance")
async def set_attendance(data: AttendanceIn, user: dict = Depends(professor_only)):
    await owned_student(data.student_id, user)
    if data.status == "none":
        await db.attendance.delete_one({"date": data.date, "student_id": data.student_id})
    else:
        await db.attendance.update_one(
            {"date": data.date, "student_id": data.student_id},
            {"$set": {"status": data.status, "professor_id": user["id"], "updated_at": now_iso()}},
            upsert=True)
    return {"ok": True}


@api.get("/attendance/report")
async def report(month: str = Query(pattern=r"^\d{4}-\d{2}$"),
                 professor_id: Optional[str] = None,
                 user: dict = Depends(current_user)):
    role = user["role"]
    if role == "admin":
        sfilter: dict = {} if not professor_id else {"professor_id": professor_id}
    elif role == "professor":
        sfilter = {"professor_id": user["id"]}
    else:
        sfilter = {"id": user.get("student_id")}
    students = await db.students.find(sfilter, {"_id": 0, "id": 1, "full_name": 1}).sort("full_name", 1).to_list(5000)
    sids = [s["id"] for s in students]
    docs = await db.attendance.find(
        {"date": {"$regex": f"^{re.escape(month)}-"}, "student_id": {"$in": sids}}, {"_id": 0}).to_list(50000)
    stats = {s["id"]: {"student_id": s["id"], "name": s["full_name"], "present": 0, "absent": 0, "days": []}
             for s in students}
    dates = set()
    for d in docs:
        dates.add(d["date"])
        st = stats.get(d["student_id"])
        if st and d["status"] in ("present", "absent"):
            st[d["status"]] += 1
            st["days"].append({"date": d["date"], "status": d["status"]})
    for st in stats.values():
        st["days"].sort(key=lambda x: x["date"])
    return {"month": month, "class_days": len(dates), "students": list(stats.values())}


# ---------- Schedule ----------
@api.get("/schedule")
async def get_schedule(user: dict = Depends(current_user)):
    pid = scope_pid(user)
    if pid is None:
        return [{"weekday": i, "active": False, "start": "", "end": ""} for i in range(7)]
    docs = {d["weekday"]: d for d in await db.schedule.find({"professor_id": pid}, {"_id": 0}).to_list(7)}
    return [docs.get(i, {"weekday": i, "active": False, "start": "", "end": ""}) for i in range(7)]


@api.put("/schedule/{weekday}")
async def set_schedule(weekday: int, data: ScheduleIn, user: dict = Depends(professor_only)):
    if weekday < 0 or weekday > 6:
        raise HTTPException(status_code=422, detail="Dia inválido.")
    if data.active and (not data.start or not data.end):
        raise HTTPException(status_code=422, detail="Informe o horário de início e fim.")
    doc = {"professor_id": user["id"], "weekday": weekday, **data.model_dump()}
    await db.schedule.update_one({"professor_id": user["id"], "weekday": weekday}, {"$set": doc}, upsert=True)
    return doc


# ---------- Messages ----------
@api.get("/messages")
async def list_messages(limit: int = Query(100, ge=1, le=300), user: dict = Depends(current_user)):
    pid = scope_pid(user)
    q: dict = {} if pid is None else {"professor_id": pid}
    docs = await db.messages.find(q, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
    return list(reversed(docs))


@api.post("/messages")
async def send_message(data: MessageIn, user: dict = Depends(current_user)):
    pid = scope_pid(user)
    if pid is None:
        raise HTTPException(status_code=403, detail="O Mestre não envia avisos de cidade.")
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Mensagem vazia.")
    doc = {"id": str(uuid.uuid4()), "text": text, "sender_id": user["id"],
           "sender_name": user.get("name") or user["username"], "role": user["role"],
           "professor_id": pid, "created_at": now_iso()}
    await db.messages.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.delete("/messages/{message_id}")
async def delete_message(message_id: str, user: dict = Depends(professor_only)):
    res = await db.messages.delete_one({"id": message_id, "professor_id": user["id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Mensagem não encontrada.")
    return {"ok": True}


app.include_router(api)
app.add_middleware(CORSMiddleware, allow_credentials=False, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.on_event("startup")
async def startup():
    await seed_admin()
    app.state.index_task = asyncio.create_task(ensure_indexes())


@app.on_event("shutdown")
async def shutdown():
    app.state.index_task.cancel()
    with suppress(asyncio.CancelledError):
        await app.state.index_task
    client.close()
