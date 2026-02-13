from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic import EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import json
import re
import xml.etree.ElementTree as ET
from html import unescape
from urllib.request import urlopen, Request
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
RUPEE_SYMBOL = "\u20B9"

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
    transaction_type: str = "debit"
    sentiment: Optional[str] = None
    ref_id: Optional[str] = None
    merchant_key: Optional[str] = None
    upi_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TransactionCreate(BaseModel):
    user_id: str
    amount: float
    description: str
    date: Optional[datetime] = None
    transaction_type: Optional[str] = None

class SMSTransactionRequest(BaseModel):
    user_id: str
    sms_text: str
    date: Optional[datetime] = None

class TransactionCategoryUpdate(BaseModel):
    user_id: str
    category: str
    apply_to_similar: bool = False

class TransactionAmountUpdate(BaseModel):
    user_id: str
    amount: float

class TransactionUpdateRequest(BaseModel):
    user_id: str
    amount: Optional[float] = None
    category: Optional[str] = None
    transaction_type: Optional[str] = None
    description: Optional[str] = None
    date: Optional[datetime] = None


class TransactionSimilarityRequest(BaseModel):
    user_id: str


class TransactionSimilarityResponse(BaseModel):
    match_count: int
    merchant_key: Optional[str] = None
    upi_id: Optional[str] = None
    sample_descriptions: List[str] = []

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

class FinancialNewsItem(BaseModel):
    title: str
    summary: str
    source: str
    link: str
    published_at: Optional[datetime] = None
    sentiment: str = "neutral"

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

def normalize_transaction_type(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in {"credit", "debit"}:
        raise HTTPException(status_code=400, detail="transaction_type must be 'credit' or 'debit'")
    return normalized

def infer_transaction_type(text: str) -> str:
    lowered = (text or "").lower()

    credit_keywords = [
        "credited",
        "credit alert",
        "received",
        "salary",
        "refund",
        "cashback",
        "cash back",
        "deposited",
        "deposit",
        "interest credited",
        "reversal",
    ]
    debit_keywords = [
        "debited",
        "spent",
        "purchase",
        "withdrawn",
        "paid",
        "bill",
        "emi",
        "sent",
        "transfer to",
    ]

    if any(keyword in lowered for keyword in credit_keywords):
        return "credit"
    if any(keyword in lowered for keyword in debit_keywords):
        return "debit"
    return "debit"


def _normalize_merchant_label(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9@._ -]", " ", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned[:80]


def _extract_ref_id(text: str) -> Optional[str]:
    sms = text or ""
    patterns = [
        r"(?:utr|rrn|ref(?:erence)?(?:\s*no)?|txn(?:\s*id)?|transaction\s*id)\s*[:\-]?\s*([A-Za-z0-9]{6,30})",
        r"\b([A-Z0-9]{10,24})\b",
    ]

    candidates: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sms, flags=re.IGNORECASE):
            value = (match.group(1) or "").strip().upper()
            if len(value) >= 6:
                candidates.append(value)
        if candidates:
            break

    if not candidates:
        return None

    return sorted(candidates, key=len, reverse=True)[0]


def _extract_upi_id(text: str) -> Optional[str]:
    sms = text or ""
    match = re.search(r"\b([a-zA-Z0-9.\-_]{2,})@([a-zA-Z]{2,})\b", sms)
    if not match:
        return None
    return f"{match.group(1).lower()}@{match.group(2).lower()}"


def _extract_merchant_key(text: str) -> Optional[str]:
    sms = text or ""
    upi_id = _extract_upi_id(sms)
    if upi_id:
        return f"upi:{upi_id}"

    patterns = [
        r"(?:to|at|for|towards)\s+([A-Za-z0-9&._ -]{3,40})",
        r"(?:merchant|payee)\s*[:\-]?\s*([A-Za-z0-9&._ -]{3,40})",
        r"(?:on)\s+([A-Za-z0-9&._ -]{3,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, sms, flags=re.IGNORECASE)
        if match:
            merchant = _normalize_merchant_label(match.group(1))
            if merchant:
                return f"merchant:{merchant}"

    fallback = _normalize_merchant_label(sms[:40])
    return f"merchant:{fallback}" if fallback else None


def _build_similar_match_query(
    user_id: str,
    base_transaction_id: str,
    merchant_key: Optional[str],
    upi_id: Optional[str],
) -> Dict[str, Any]:
    query: Dict[str, Any] = {"user_id": user_id, "id": {"$ne": base_transaction_id}}

    or_conditions: List[Dict[str, Any]] = []
    if upi_id:
        escaped_upi = re.escape(upi_id)
        or_conditions.append({"upi_id": upi_id})
        or_conditions.append({"description": {"$regex": escaped_upi, "$options": "i"}})

    if merchant_key:
        merchant_label = merchant_key.replace("merchant:", "", 1).strip()
        or_conditions.append({"merchant_key": merchant_key})
        if merchant_label:
            escaped_label = re.escape(merchant_label)
            or_conditions.append(
                {"description": {"$regex": escaped_label, "$options": "i"}}
            )

    if or_conditions:
        query["$or"] = or_conditions

    return query

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
Total: {RUPEE_SYMBOL}{total_spending:.2f}
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
Total: {RUPEE_SYMBOL}{total_spending:.2f}
Average: {RUPEE_SYMBOL}{avg_spending:.2f}
Most recent: {latest_desc}

Provide 2 concise, actionable insights for this category.
"""

    return await invoke_llm("You are a financial advisor.", prompt)

def _extract_summary(title: str, text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = unescape(re.sub(r"\s+", " ", cleaned)).strip()

    words = cleaned.split()
    if len(words) > 100:
        return " ".join(words[:100]).strip()

    if len(words) < 50:
        title_words = re.sub(r"\s+", " ", title or "").strip().split()
        combined = title_words + words
        filler = (
            "Market participants are evaluating near term impact, and analysts are watching "
            "earnings expectations, policy direction, and risk sentiment for potential shifts."
        ).split()
        while len(combined) < 50:
            combined.extend(filler)
        return " ".join(combined[:100]).strip()

    return " ".join(words[:100]).strip()

def _guess_sentiment(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    positive_keywords = [
        "rally",
        "gain",
        "up",
        "growth",
        "profit",
        "record high",
        "beat",
        "strong",
        "surge",
    ]
    negative_keywords = [
        "drop",
        "down",
        "fall",
        "loss",
        "cut",
        "weak",
        "concern",
        "inflation risk",
        "slump",
    ]

    positive_score = sum(1 for word in positive_keywords if word in text)
    negative_score = sum(1 for word in negative_keywords if word in text)

    if positive_score > negative_score:
        return "positive"
    if negative_score > positive_score:
        return "negative"
    return "neutral"

def _parse_rss_items(xml_data: bytes, default_source: str) -> List[FinancialNewsItem]:
    root = ET.fromstring(xml_data)
    items: List[FinancialNewsItem] = []

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        source = default_source

        source_node = item.find("source")
        if source_node is not None and (source_node.text or "").strip():
            source = source_node.text.strip()

        pub_date_raw = (item.findtext("pubDate") or "").strip()
        published_at = None
        if pub_date_raw:
            try:
                published_at = datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                published_at = None

        if not title or not link:
            continue

        summary = _extract_summary(title, description)
        items.append(
            FinancialNewsItem(
                title=title,
                summary=summary,
                source=source,
                link=link,
                published_at=published_at,
                sentiment=_guess_sentiment(title, summary),
            )
        )

    return items


def _format_inr(amount: float) -> str:
    return f"{RUPEE_SYMBOL}{float(amount):,.2f}"


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return datetime.utcnow()
    return datetime.utcnow()


def _transaction_datetime(item: Dict[str, Any]) -> datetime:
    return _as_datetime(item.get("date") or item.get("created_at"))


def _enforce_rupee_only(text: str) -> str:
    if not text:
        return text

    normalized = re.sub(r"\bUSD\b", "INR", text, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        fr"{RUPEE_SYMBOL}\1",
        normalized,
    )
    normalized = re.sub(r"\bRs\.?\s*", RUPEE_SYMBOL, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdollars?\b", "rupees", normalized, flags=re.IGNORECASE)
    return normalized


def _is_data_question(message: str) -> bool:
    lowered = (message or "").lower()
    keywords = [
        "spend",
        "spent",
        "expense",
        "expenses",
        "transaction",
        "transactions",
        "category",
        "categories",
        "balance",
        "credit",
        "debit",
        "income",
        "cashflow",
        "cash flow",
        "monthly",
        "month",
        "today",
        "week",
        "summary",
        "analytics",
        "report",
    ]
    return any(word in lowered for word in keywords)


async def _tool_build_financial_snapshot(user_id: str) -> Dict[str, Any]:
    transactions = await db.transactions.find({"user_id": user_id}).sort("date", -1).limit(500).to_list(500)

    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)

    total_debit = 0.0
    total_credit = 0.0
    month_debit = 0.0
    month_credit = 0.0
    today_debit = 0.0
    today_credit = 0.0
    category_totals: Dict[str, float] = {}
    recent_items: List[Dict[str, str]] = []

    for index, item in enumerate(transactions):
        amount = float(item.get("amount", 0) or 0)
        tx_type = str(item.get("transaction_type", "debit")).lower()
        tx_date = _transaction_datetime(item)
        category = str(item.get("category", "Other"))

        if tx_type == "credit":
            total_credit += amount
            if tx_date >= month_start:
                month_credit += amount
            if tx_date.date() == now.date():
                today_credit += amount
        else:
            total_debit += amount
            if tx_date >= month_start:
                month_debit += amount
            if tx_date.date() == now.date():
                today_debit += amount
            category_totals[category] = category_totals.get(category, 0) + amount

        if index < 5:
            recent_items.append(
                {
                    "date": tx_date.strftime("%d %b %Y"),
                    "category": category,
                    "amount": _format_inr(amount),
                    "type": tx_type,
                    "description": str(item.get("description", ""))[:80],
                }
            )

    top_categories = sorted(category_totals.items(), key=lambda pair: pair[1], reverse=True)[:5]

    return {
        "transaction_count": len(transactions),
        "total_debit": total_debit,
        "total_credit": total_credit,
        "net_cashflow": total_credit - total_debit,
        "month_debit": month_debit,
        "month_credit": month_credit,
        "today_debit": today_debit,
        "today_credit": today_credit,
        "top_categories": top_categories,
        "recent_transactions": recent_items,
    }


def _build_data_answer(message: str, snapshot: Dict[str, Any]) -> str:
    if snapshot["transaction_count"] == 0:
        return (
            "I could not find any transactions yet. "
            f"Once your transactions are added, I can answer data questions in rupees ({RUPEE_SYMBOL})."
        )

    lowered = (message or "").lower()
    lines = [
        "Here is your latest financial data:",
        f"Total debit: {_format_inr(snapshot['total_debit'])}",
        f"Total credit: {_format_inr(snapshot['total_credit'])}",
        f"Net cashflow: {_format_inr(snapshot['net_cashflow'])}",
        f"This month debit: {_format_inr(snapshot['month_debit'])}",
        f"This month credit: {_format_inr(snapshot['month_credit'])}",
    ]

    if "today" in lowered:
        lines.append(f"Today debit: {_format_inr(snapshot['today_debit'])}")
        lines.append(f"Today credit: {_format_inr(snapshot['today_credit'])}")

    if any(word in lowered for word in ["category", "categories", "top"]):
        if snapshot["top_categories"]:
            lines.append("Top spending categories:")
            for category, amount in snapshot["top_categories"]:
                lines.append(f"- {category}: {_format_inr(amount)}")

    if any(word in lowered for word in ["recent", "last", "latest", "transaction"]):
        if snapshot["recent_transactions"]:
            lines.append("Recent transactions:")
            for item in snapshot["recent_transactions"]:
                sign = "+" if item["type"] == "credit" else "-"
                lines.append(
                    f"- {item['date']} | {item['category']} | {sign}{item['amount']} | {item['description']}"
                )

    return "\n".join(lines)

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
    trans_dict["transaction_type"] = (
        normalize_transaction_type(transaction.transaction_type)
        if transaction.transaction_type
        else infer_transaction_type(transaction.description)
    )
    trans_dict["upi_id"] = _extract_upi_id(transaction.description)
    trans_dict["merchant_key"] = _extract_merchant_key(transaction.description)

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

    transaction_date = request.date or datetime.utcnow()
    description = request.sms_text[:100]
    ref_id = _extract_ref_id(request.sms_text)
    upi_id = _extract_upi_id(request.sms_text)
    merchant_key = _extract_merchant_key(request.sms_text)

    if ref_id:
        existing_by_ref = await db.transactions.find_one(
            {
                "user_id": request.user_id,
                "source": "sms",
                "ref_id": ref_id,
            }
        )
        if existing_by_ref:
            return Transaction(**existing_by_ref)

    start_window = transaction_date - timedelta(minutes=3)
    end_window = transaction_date + timedelta(minutes=3)
    fallback_query: Dict[str, Any] = {
        "user_id": request.user_id,
        "source": "sms",
        "amount": amount,
        "date": {"$gte": start_window, "$lte": end_window},
    }
    if merchant_key:
        fallback_query["$or"] = [
            {"merchant_key": merchant_key},
            {"description": description},
        ]
    else:
        fallback_query["description"] = description

    existing_sms_transaction = await db.transactions.find_one(fallback_query)
    if existing_sms_transaction:
        return Transaction(**existing_sms_transaction)

    trans_obj = Transaction(
        user_id=request.user_id,
        amount=amount,
        category=ai_result["category"],
        description=description,
        date=transaction_date,
        source="sms",
        transaction_type=infer_transaction_type(request.sms_text),
        sentiment=ai_result["sentiment"],
        ref_id=ref_id,
        merchant_key=merchant_key,
        upi_id=upi_id,
    )

    await db.transactions.insert_one(trans_obj.dict())
    return trans_obj


@api_router.post(
    "/transactions/{transaction_id}/similar-preview",
    response_model=TransactionSimilarityResponse,
)
async def preview_similar_transactions(
    transaction_id: str, request: TransactionSimilarityRequest
):
    selected_transaction = await db.transactions.find_one(
        {"id": transaction_id, "user_id": request.user_id}
    )
    if not selected_transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    merchant_key = selected_transaction.get("merchant_key") or _extract_merchant_key(
        selected_transaction.get("description", "")
    )
    upi_id = selected_transaction.get("upi_id") or _extract_upi_id(
        selected_transaction.get("description", "")
    )

    if not merchant_key and not upi_id:
        return TransactionSimilarityResponse(
            match_count=0, merchant_key=None, upi_id=None, sample_descriptions=[]
        )

    query = _build_similar_match_query(
        user_id=request.user_id,
        base_transaction_id=transaction_id,
        merchant_key=merchant_key,
        upi_id=upi_id,
    )

    similar_transactions = await db.transactions.find(query).sort("date", -1).limit(5).to_list(5)
    match_count = await db.transactions.count_documents(query)

    return TransactionSimilarityResponse(
        match_count=match_count,
        merchant_key=merchant_key,
        upi_id=upi_id,
        sample_descriptions=[str(item.get("description", ""))[:80] for item in similar_transactions],
    )

@api_router.get("/transactions/{user_id}", response_model=List[Transaction])
async def get_user_transactions(user_id: str, limit: int = 50):
    raw_transactions = await db.transactions.find({"user_id": user_id}).to_list(2000)

    normalized: List[Dict[str, Any]] = []
    for item in raw_transactions:
        doc = dict(item)
        doc["date"] = doc.get("date") or doc.get("created_at") or datetime.utcnow()
        normalized.append(doc)

    normalized.sort(key=lambda tx: _transaction_datetime(tx), reverse=True)
    return [Transaction(**t) for t in normalized[: max(1, limit)]]

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

    if request.apply_to_similar and updated_transaction:
        merchant_key = updated_transaction.get("merchant_key") or _extract_merchant_key(
            updated_transaction.get("description", "")
        )
        upi_id = updated_transaction.get("upi_id") or _extract_upi_id(
            updated_transaction.get("description", "")
        )

        bulk_query = _build_similar_match_query(
            user_id=request.user_id,
            base_transaction_id=transaction_id,
            merchant_key=merchant_key,
            upi_id=upi_id,
        )

        if bulk_query.get("$or"):
            await db.transactions.update_many(
                bulk_query,
                {"$set": {"category": normalized_category}},
            )

    return Transaction(**updated_transaction)

@api_router.put("/transactions/{transaction_id}/amount", response_model=Transaction)
async def update_transaction_amount(transaction_id: str, request: TransactionAmountUpdate):
    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than 0")

    update_result = await db.transactions.update_one(
        {"id": transaction_id, "user_id": request.user_id},
        {"$set": {"amount": float(request.amount)}},
    )

    if update_result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")

    updated_transaction = await db.transactions.find_one(
        {"id": transaction_id, "user_id": request.user_id}
    )
    return Transaction(**updated_transaction)

@api_router.put("/transactions/{transaction_id}", response_model=Transaction)
async def update_transaction(transaction_id: str, request: TransactionUpdateRequest):
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

    update_fields: Dict[str, object] = {}

    if request.amount is not None:
        if request.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        update_fields["amount"] = float(request.amount)

    if request.category is not None:
        normalized_category = request.category.strip().title()
        if normalized_category not in allowed_categories:
            raise HTTPException(status_code=400, detail="Invalid category")
        update_fields["category"] = normalized_category

    if request.transaction_type is not None:
        update_fields["transaction_type"] = normalize_transaction_type(
            request.transaction_type
        )

    if request.description is not None:
        cleaned_description = request.description.strip()
        if not cleaned_description:
            raise HTTPException(status_code=400, detail="Description cannot be empty")
        update_fields["description"] = cleaned_description
        update_fields["upi_id"] = _extract_upi_id(cleaned_description)
        update_fields["merchant_key"] = _extract_merchant_key(cleaned_description)

    if request.date is not None:
        update_fields["date"] = request.date

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    update_result = await db.transactions.update_one(
        {"id": transaction_id, "user_id": request.user_id},
        {"$set": update_fields},
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
    raw_transactions = await db.transactions.find({"user_id": user_id}).to_list(3000)
    transactions = [
        t for t in raw_transactions if _transaction_datetime(t) >= start_date
    ]

    total_debit = sum(
        t.get("amount", 0)
        for t in transactions
        if t.get("transaction_type", "debit") == "debit"
    )
    total_credit = sum(
        t.get("amount", 0)
        for t in transactions
        if t.get("transaction_type", "debit") == "credit"
    )
    total_spending = total_debit

    categories: Dict[str, float] = {}
    for t in transactions:
        if t.get("transaction_type", "debit") != "debit":
            continue
        cat = t.get("category", "Other")
        categories[cat] = categories.get(cat, 0) + t.get("amount", 0)

    daily_spending: Dict[str, float] = {}
    for t in transactions:
        if t.get("transaction_type", "debit") != "debit":
            continue
        date_key = _transaction_datetime(t).strftime("%Y-%m-%d")
        daily_spending[date_key] = daily_spending.get(date_key, 0) + t.get("amount", 0)

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for t in transactions:
        sent = t.get("sentiment", "neutral")
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

    return {
        "total_spending": total_spending,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "transaction_count": len(transactions),
        "average_transaction": (
            total_debit / max(1, len([t for t in transactions if t.get("transaction_type", "debit") == "debit"]))
        ),
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

@api_router.get("/news/financial", response_model=List[FinancialNewsItem])
async def get_financial_news(limit: int = 10):
    rss_sources = [
        ("https://feeds.reuters.com/reuters/businessNews", "Reuters Business"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US", "Yahoo Finance"),
        ("https://www.investing.com/rss/news_25.rss", "Investing.com"),
    ]

    aggregated: List[FinancialNewsItem] = []

    for url, source_name in rss_sources:
        try:
            request = Request(
                url,
                headers={"User-Agent": "SpendWiseNewsBot/1.0"},
            )
            with urlopen(request, timeout=8) as response:
                xml_data = response.read()
            aggregated.extend(_parse_rss_items(xml_data, source_name))
        except Exception as error:
            logging.warning(f"Failed to fetch financial news from {source_name}: {error}")

    if not aggregated:
        raise HTTPException(status_code=503, detail="Unable to fetch financial news right now")

    dedup: Dict[str, FinancialNewsItem] = {}
    for item in aggregated:
        if item.link not in dedup:
            dedup[item.link] = item

    sorted_items = sorted(
        dedup.values(),
        key=lambda item: item.published_at.timestamp() if item.published_at else 0,
        reverse=True,
    )

    return sorted_items[: max(1, min(limit, 25))]

@api_router.post("/chat")
async def chat_with_ai(request: ChatRequest):
    user_msg = ChatMessage(
        user_id=request.user_id,
        role="user",
        message=request.message,
    )
    await db.chat_messages.insert_one(user_msg.dict())

    snapshot = await _tool_build_financial_snapshot(request.user_id)

    if _is_data_question(request.message):
        response = _build_data_answer(request.message, snapshot)
    else:
        categories_context = ", ".join(
            f"{category}: {_format_inr(amount)}"
            for category, amount in snapshot["top_categories"]
        ) or "No category data yet"

        context = (
            "User financial tool output:\n"
            f"- Transactions tracked: {snapshot['transaction_count']}\n"
            f"- Total debit: {_format_inr(snapshot['total_debit'])}\n"
            f"- Total credit: {_format_inr(snapshot['total_credit'])}\n"
            f"- Net cashflow: {_format_inr(snapshot['net_cashflow'])}\n"
            f"- This month debit: {_format_inr(snapshot['month_debit'])}\n"
            f"- This month credit: {_format_inr(snapshot['month_credit'])}\n"
            f"- Top categories: {categories_context}\n"
            f"Important: Always use Indian rupee symbol ({RUPEE_SYMBOL}). Never use $ or USD."
        )

        response = await invoke_llm(
            (
                "You are a helpful financial advisor for an Indian user. "
                f"Use only rupee symbol ({RUPEE_SYMBOL}) for all currency values. "
                "If user asks about personal financial data, rely on provided tool output."
            ),
            f"{context}\n\nUser question: {request.message}",
        )

    response = _enforce_rupee_only(response)

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
