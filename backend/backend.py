from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic import EmailStr
from typing import List, Optional, Dict
import uuid
from datetime import datetime, timedelta
import json
import re
from passlib.context import CryptContext

from openai import AsyncOpenAI

# ==================== INIT ====================

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

app = FastAPI()
api_router = APIRouter(prefix="/api")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==================== LLM HELPER ====================

async def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    response = await groq_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()

# ==================== MODELS ====================

class Transaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    amount: float
    category: str
    description: str
    date: datetime
    source: str
    sentiment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TransactionCreate(BaseModel):
    user_id: str
    amount: float
    description: str
    date: Optional[datetime] = None

class SMSTransactionRequest(BaseModel):
    user_id: str
    sms_text: str
    date: Optional[datetime] = None

class TransactionCategoryUpdate(BaseModel):
    user_id: str
    category: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: EmailStr
    phone: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class Credit(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    credit_score: Optional[int] = None
    card_name: str
    card_balance: float
    credit_limit: float
    payment_due_date: Optional[datetime] = None
    utilization: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CreditCreate(BaseModel):
    user_id: str
    credit_score: Optional[int] = None
    card_name: str
    card_balance: float
    credit_limit: float
    payment_due_date: Optional[datetime] = None

class Habit(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    goal: str
    target_amount: float
    current_amount: float = 0.0
    category: str
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    status: str = "active"
    progress: float = 0.0

class HabitCreate(BaseModel):
    user_id: str
    goal: str
    target_amount: float
    category: str
    end_date: Optional[datetime] = None

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    role: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ==================== AI FUNCTIONS ====================

async def categorize_transaction_with_ai(text: str, amount: float) -> Dict[str, str]:
    try:
        prompt = f"""
Analyze transaction:
Amount: ${amount}
Description: {text}

Return ONLY JSON:
{{
 "category":"Food|Transport|Shopping|Bills|Entertainment|Health|Education|Travel|Other",
 "sentiment":"positive|neutral|negative"
}}
"""

        response = await invoke_llm(
            "You categorize financial transactions.",
            prompt,
            temperature=0.1,
        )

        cleaned = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)

    except Exception as e:
        logging.error(e)
        return {"category": "Other", "sentiment": "neutral"}

async def generate_insights(user_id: str) -> str:
    transactions = await db.transactions.find(
        {"user_id": user_id}
    ).sort("date", -1).limit(20).to_list(20)

    if not transactions:
        return "Start tracking expenses to see insights."

    total_spending = sum(t.get("amount", 0) for t in transactions)

    categories = {}
    for t in transactions:
        cat = t.get("category", "Other")
        categories[cat] = categories.get(cat, 0) + t.get("amount", 0)

    prompt = f"""
User spending:
Total: ${total_spending:.2f}
Categories: {categories}

Provide 3 financial tips.
"""

    return await invoke_llm("You are a financial advisor.", prompt)

async def generate_category_insights(user_id: str, category: str) -> str:
    transactions = await db.transactions.find(
        {"user_id": user_id, "category": category}
    ).sort("date", -1).limit(15).to_list(15)

    if not transactions:
        return f"No {category} transactions yet. Add a few to unlock insights."

    total_spending = sum(t.get("amount", 0) for t in transactions)
    avg_spending = total_spending / len(transactions) if transactions else 0
    latest = transactions[0]
    latest_desc = latest.get("description", "recent transaction")[:80]

    prompt = f"""
Category: {category}
Recent count: {len(transactions)}
Total: ${total_spending:.2f}
Average: ${avg_spending:.2f}
Most recent: {latest_desc}

Provide 2 concise, actionable insights for this category.
"""

    return await invoke_llm("You are a financial advisor.", prompt)

# ==================== ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "Financial Habit Tracker API"}

# User endpoints
@api_router.post("/users", response_model=User)
async def create_user(user: UserCreate):
    existing_user = await db.users.find_one({"email": user.email.lower()})
    if existing_user:
        raise HTTPException(status_code=409, detail="Email already exists")

    user_data = user.dict()
    user_data["email"] = user_data["email"].lower()
    user_obj = User(**user_data)
    await db.users.insert_one(user_obj.dict())
    return user_obj

@api_router.post("/auth/signup", response_model=User)
async def signup(request: SignupRequest):
    if request.password != request.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    normalized_email = request.email.lower()
    existing_user = await db.users.find_one({"email": normalized_email})
    if existing_user:
        raise HTTPException(status_code=409, detail="Email already exists")

    user_obj = User(
        name=request.name.strip(),
        email=normalized_email,
    )

    user_doc = user_obj.dict()
    user_doc["password_hash"] = pwd_context.hash(request.password)
    await db.users.insert_one(user_doc)
    return user_obj

@api_router.post("/auth/login", response_model=User)
async def login(request: LoginRequest):
    normalized_email = request.email.lower()
    existing_user = await db.users.find_one({"email": normalized_email})

    if not existing_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    password_hash = existing_user.get("password_hash")
    if not password_hash or not pwd_context.verify(request.password, password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return User(**existing_user)

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await db.users.find_one({"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user)

@api_router.post("/transactions/manual", response_model=Transaction)
async def create_manual_transaction(transaction: TransactionCreate):
    ai_result = await categorize_transaction_with_ai(
        transaction.description,
        transaction.amount,
    )

    trans_dict = transaction.dict()
    trans_dict["date"] = trans_dict.get("date") or datetime.utcnow()
    trans_dict["source"] = "manual"
    trans_dict["category"] = ai_result["category"]
    trans_dict["sentiment"] = ai_result["sentiment"]

    trans_obj = Transaction(**trans_dict)
    await db.transactions.insert_one(trans_obj.dict())
    return trans_obj

@api_router.post("/transactions/sms", response_model=Transaction)
async def create_sms_transaction(request: SMSTransactionRequest):
    amount_match = re.search(
        r"(?:Rs\.?|INR|\$)\s*([0-9,]+\.?[0-9]*)",
        request.sms_text,
        re.IGNORECASE,
    )

    if not amount_match:
        raise HTTPException(400, "Amount not found")

    amount = float(amount_match.group(1).replace(",", ""))

    ai_result = await categorize_transaction_with_ai(
        request.sms_text,
        amount,
    )

    trans_obj = Transaction(
        user_id=request.user_id,
        amount=amount,
        category=ai_result["category"],
        description=request.sms_text[:100],
        date=request.date or datetime.utcnow(),
        source="sms",
        sentiment=ai_result["sentiment"],
    )

    await db.transactions.insert_one(trans_obj.dict())
    return trans_obj

@api_router.get("/transactions/{user_id}", response_model=List[Transaction])
async def get_user_transactions(user_id: str, limit: int = 50):
    transactions = await db.transactions.find(
        {"user_id": user_id}
    ).sort("date", -1).limit(limit).to_list(limit)
    return [Transaction(**t) for t in transactions]

@api_router.put("/transactions/{transaction_id}/category", response_model=Transaction)
async def update_transaction_category(transaction_id: str, request: TransactionCategoryUpdate):
    allowed_categories = {
        "Food",
        "Transport",
        "Shopping",
        "Bills",
        "Entertainment",
        "Health",
        "Education",
        "Travel",
        "Other",
    }

    normalized_category = request.category.strip().title()
    if normalized_category not in allowed_categories:
        raise HTTPException(status_code=400, detail="Invalid category")

    update_result = await db.transactions.update_one(
        {"id": transaction_id, "user_id": request.user_id},
        {"$set": {"category": normalized_category}},
    )

    if update_result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")

    updated_transaction = await db.transactions.find_one(
        {"id": transaction_id, "user_id": request.user_id}
    )
    return Transaction(**updated_transaction)

@api_router.get("/transactions/{user_id}/analytics")
async def get_transaction_analytics(user_id: str, days: int = 30):
    start_date = datetime.utcnow() - timedelta(days=days)

    transactions = await db.transactions.find(
        {"user_id": user_id, "date": {"$gte": start_date}}
    ).to_list(1000)

    total_spending = sum(t.get("amount", 0) for t in transactions)

    categories: Dict[str, float] = {}
    for t in transactions:
        cat = t.get("category", "Other")
        categories[cat] = categories.get(cat, 0) + t.get("amount", 0)

    daily_spending: Dict[str, float] = {}
    for t in transactions:
        date_key = t.get("date", datetime.utcnow()).strftime("%Y-%m-%d")
        daily_spending[date_key] = daily_spending.get(date_key, 0) + t.get("amount", 0)

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for t in transactions:
        sent = t.get("sentiment", "neutral")
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

    return {
        "total_spending": total_spending,
        "transaction_count": len(transactions),
        "average_transaction": total_spending / len(transactions) if transactions else 0,
        "categories": categories,
        "daily_spending": daily_spending,
        "sentiment": sentiment_counts,
    }

# Credit endpoints
@api_router.post("/credits", response_model=Credit)
async def create_credit(credit: CreditCreate):
    credit_dict = credit.dict()
    credit_dict["utilization"] = (
        credit_dict["card_balance"] / credit_dict["credit_limit"] * 100
        if credit_dict["credit_limit"] > 0
        else 0
    )
    credit_obj = Credit(**credit_dict)
    await db.credits.insert_one(credit_obj.dict())
    return credit_obj

@api_router.get("/credits/{user_id}", response_model=List[Credit])
async def get_user_credits(user_id: str):
    credits = await db.credits.find({"user_id": user_id}).to_list(100)
    return [Credit(**c) for c in credits]

@api_router.put("/credits/{credit_id}", response_model=Credit)
async def update_credit(credit_id: str, credit: CreditCreate):
    credit_dict = credit.dict()
    credit_dict["utilization"] = (
        credit_dict["card_balance"] / credit_dict["credit_limit"] * 100
        if credit_dict["credit_limit"] > 0
        else 0
    )

    await db.credits.update_one({"id": credit_id}, {"$set": credit_dict})
    updated_credit = await db.credits.find_one({"id": credit_id})
    if not updated_credit:
        raise HTTPException(status_code=404, detail="Credit not found")
    return Credit(**updated_credit)

# Habit endpoints
@api_router.post("/habits", response_model=Habit)
async def create_habit(habit: HabitCreate):
    habit_obj = Habit(**habit.dict())
    await db.habits.insert_one(habit_obj.dict())
    return habit_obj

@api_router.get("/habits/{user_id}", response_model=List[Habit])
async def get_user_habits(user_id: str):
    habits = await db.habits.find({"user_id": user_id}).to_list(100)
    return [Habit(**h) for h in habits]

async def update_habit_progress(user_id: str, category: str, amount: float):
    habits = await db.habits.find(
        {"user_id": user_id, "category": category, "status": "active"}
    ).to_list(100)

    for habit in habits:
        new_amount = habit.get("current_amount", 0) + amount
        progress = (new_amount / habit.get("target_amount", 1)) * 100

        status = "active"
        if progress >= 100:
            status = "completed"

        await db.habits.update_one(
            {"id": habit["id"]},
            {
                "$set": {
                    "current_amount": new_amount,
                    "progress": min(progress, 100),
                    "status": status,
                }
            },
        )

@api_router.get("/insights/{user_id}")
async def get_ai_insights(user_id: str):
    insights = await generate_insights(user_id)
    return {"insights": insights}

@api_router.get("/insights/{user_id}/category/{category}")
async def get_category_insights(user_id: str, category: str):
    insights = await generate_category_insights(user_id, category)
    return {"insights": insights}

@api_router.post("/chat")
async def chat_with_ai(request: ChatRequest):
    user_msg = ChatMessage(
        user_id=request.user_id,
        role="user",
        message=request.message,
    )
    await db.chat_messages.insert_one(user_msg.dict())

    transactions = await db.transactions.find(
        {"user_id": request.user_id}
    ).limit(10).to_list(10)

    total_spending = sum(t.get("amount", 0) for t in transactions)

    context = f"User recent spending total: ${total_spending:.2f}"

    response = await invoke_llm(
        f"You are a helpful financial advisor. {context}",
        request.message,
    )

    assistant_msg = ChatMessage(
        user_id=request.user_id,
        role="assistant",
        message=response,
    )
    await db.chat_messages.insert_one(assistant_msg.dict())

    return {"response": response}

@api_router.get("/chat/{user_id}", response_model=List[ChatMessage])
async def get_chat_history(user_id: str, limit: int = 50):
    messages = await db.chat_messages.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    messages.reverse()
    return [ChatMessage(**m) for m in messages]

# ==================== APP SETUP ====================

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
