from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid
from datetime import datetime, timedelta
import json
import re

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

class ChatRequest(BaseModel):
    user_id: str
    message: str

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

# ==================== ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "Financial Habit Tracker API"}

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

@api_router.get("/insights/{user_id}")
async def get_ai_insights(user_id: str):
    insights = await generate_insights(user_id)
    return {"insights": insights}

@api_router.post("/chat")
async def chat_with_ai(request: ChatRequest):
    transactions = await db.transactions.find(
        {"user_id": request.user_id}
    ).limit(10).to_list(10)

    total_spending = sum(t.get("amount", 0) for t in transactions)

    context = f"User recent spending total: ${total_spending:.2f}"

    response = await invoke_llm(
        f"You are a helpful financial advisor. {context}",
        request.message,
    )

    return {"response": response}

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