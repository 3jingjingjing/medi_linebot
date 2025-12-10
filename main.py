# -*- coding: utf-8 -*-

import os
import sys
import httpx
import aiohttp
from fastapi import FastAPI, Request, HTTPException
from linebot import AsyncLineBotApi, WebhookParser
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ========== 環境變數檢查 ==========

channel_secret = os.getenv("ChannelSecret", None)
channel_access_token = os.getenv("ChannelAccessToken", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# 檢查必要變數
if channel_secret is None:
    print("Error: ChannelSecret not set.")
    sys.exit(1)
if channel_access_token is None:
    print("Error: ChannelAccessToken not set.")
    sys.exit(1)
if GEMINI_API_KEY is None:
    print("Warning: GEMINI_API_KEY not set.")

# ========== 初始化 APP ==========

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "MedLineBot (HTTP Version) is running!"}

session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# ========== 1. 知識庫與 Prompt 建構 ==========

GENE_KB = [
    {
        "gene": "A 基因",
        "condition": "運動後容易肌肉受損與發炎",
        "risk": "高強度爆發或間歇訓練後，肌肉損傷與延遲性痠痛風險較高。",
        "advice": "訓練以中低強度、漸進式耐力訓練為主，拉長恢復時間，避免頻繁衝刺與過度疲勞。"
    },
    {
        "gene": "B 基因",
        "condition": "耐力表現潛力較佳，但恢復速度較慢",
        "risk": "長距離訓練後疲勞堆積較明顯，如恢復不足，較易出現過度訓練狀態。",
        "advice": "適合穩定配速長跑，每週總跑量成長幅度宜保守，固定安排休息日與低強度日。"
    },
]

def build_context_from_kb(user_query: str) -> str:
    # 簡易檢索
    hits = [item for item in GENE_KB if item["gene"] in user_query]
    if not hits:
        hits = GENE_KB # 沒對中就給全部
        
    blocks = []
    for it in hits:
        blocks.append(
            f"基因：{it['gene']}\n特徵：{it['condition']}\n風險：{it['risk']}\n建議：{it['advice']}"
        )
    return "\n\n".join(blocks)

def build_med_prompt(user_query: str) -> str:
    context = build_context_from_kb(user_query)
    return f"""你是一位運動醫學專家。
【資料庫】
{context}

【問題】{user_query}

請提供專業醫學分析(不需口語化)。
"""

def build_smart_synthesis_prompt(user_query: str, med_answer: str) -> str:
    return f"""你是一位親切的運動教練。
這是醫學報告：
{med_answer}

使用者問：「{user_query}」

請根據使用者的語氣（專業或白話），將報告轉化為適合他的建議。請用繁體中文。
"""

# ========== 2. MedGamma (Hugging Face) ==========

PRIMARY_MODEL_ID = os.getenv("MEDGAMMA_MODEL_ID", "google/medgemma-27b-text-it")
FALLBACK_MODEL_ID = "google/gemma-2-9b-it"

def call_huggingface_api(model_id: str, prompt: str) -> str:
    if not HF_API_KEY:
        return "錯誤：未設定 HF_API_KEY"

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 600, "return_full_text": False}
    }
    
    resp = httpx.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        return data[0].get("generated_text", "").strip()
    return str(data)

def call_medgamma(prompt: str) -> str:
    print(f"嘗試呼叫醫學模型: {PRIMARY_MODEL_ID}")
    try:
        return call_huggingface_api(PRIMARY_MODEL_ID, prompt)
    except Exception as e:
        print(f"主模型失敗，切換備用: {FALLBACK_MODEL_ID}")
        try:
            return call_huggingface_api(FALLBACK_MODEL_ID, prompt)
        except Exception as e2:
            return f"醫學模型暫時無法使用: {e2}"

# ========== 3. Gemini (HTTP 直連版) ==========

def call_gemini_http(prompt: str) -> str:
    """
    不使用 SDK，直接用 HTTP Post 呼叫 Gemini API
    """
    if not GEMINI_API_KEY:
        return "錯誤：未設定 GEMINI_API_KEY"

    # 直接針對 gemini-1.5-flash 的網址
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        # 發送請求
        resp = httpx.post(url, headers=headers, json=payload, timeout=60)
        
        # 如果 Key 錯誤或權限不足，這裡會直接噴 400/403
        if resp.status_code != 200:
            return f"Gemini API 錯誤 (Code: {resp.status_code}): {resp.text}"

        data = resp.json()
        
        # 解析回傳的 JSON
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip()
        except (KeyError, IndexError):
            return "Gemini 回傳了無法解析的格式。"
            
    except Exception as e:
        return f"連線發生錯誤: {str(e)}"

# ========== 主流程 ==========

def answer_user_message_auto(user_query: str) -> str:
    # 1. 問 MedGamma
    med_answer = call_medgamma(build_med_prompt(user_query))
    
    # 2. 問 Gemini (轉譯)
    final_prompt = build_smart_synthesis_prompt(user_query, med_answer)
    final_answer = call_gemini_http(final_prompt)
    
    return final_answer

# ========== Webhook ==========

@app.post("/callback")
async def handle_callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    body = body.decode("utf-8")

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessage):
            user_text = event.message.text.strip()
            
            # 執行主流程
            answer = answer_user_message_auto(user_text)

            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=answer[:2000])
            )

    return "OK"
