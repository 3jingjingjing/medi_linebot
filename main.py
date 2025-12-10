# -*- coding: utf-8 -*-

import os
import sys
import httpx
import aiohttp
import google.generativeai as genai

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

if channel_secret is None:
    print("Error: ChannelSecret not set.")
    sys.exit(1)
if channel_access_token is None:
    print("Error: ChannelAccessToken not set.")
    sys.exit(1)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not set.")

# ========== 初始化 APP ==========

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "MedLineBot is running!"}

session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# ========== 知識庫 ==========

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

def match_gene_items(user_query: str):
    hits = []
    for item in GENE_KB:
        if item["gene"] in user_query:
            hits.append(item)
    if not hits:
        hits = GENE_KB
    return hits

def build_context_from_kb(user_query: str) -> str:
    items = match_gene_items(user_query)
    blocks = []
    for it in items:
        blocks.append(
            f"基因／型別：{it['gene']}\n"
            f"體質特徵：{it['condition']}\n"
            f"風險說明：{it['risk']}\n"
            f"訓練建議：{it['advice']}"
        )
    return "\n\n".join(blocks)

# ========== 模型連線區 ==========

PRIMARY_MODEL_ID = os.getenv("MEDGAMMA_MODEL_ID", "google/medgemma-27b-text-it")
FALLBACK_MODEL_ID = "google/gemma-2-9b-it"

def call_huggingface_api(model_id: str, prompt: str) -> str:
    if not HF_API_KEY:
        return "錯誤：未設定 HUGGINGFACE_API_KEY"

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    
    resp = httpx.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    
    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        return data[0].get("generated_text", "").strip()
    return str(data)

def call_medgamma(prompt: str) -> str:
    """
    醫學模型切換邏輯
    """
    print(f"嘗試呼叫主模型: {PRIMARY_MODEL_ID}")
    try:
        return call_huggingface_api(PRIMARY_MODEL_ID, prompt)
    except Exception as e:
        print(f"主模型失敗 ({e})，切換備用模型: {FALLBACK_MODEL_ID}")
        try:
            return call_huggingface_api(FALLBACK_MODEL_ID, prompt)
        except Exception as e2:
            return f"模型暫時無法使用: {str(e2)}"

def call_gemini(prompt: str) -> str:
    """
    【關鍵修正】Gemini 多重備援機制
    嘗試順序: 1.5 Flash -> 1.5 Pro -> Pro (1.0)
    """
    if not GEMINI_API_KEY:
        return "錯誤：未設定 GEMINI_API_KEY"
    
    # 定義要嘗試的模型清單
    candidate_models = [
        "gemini-1.5-flash", 
        "gemini-1.5-flash-latest", 
        "gemini-1.5-pro", 
        "gemini-pro"
    ]
    
    for model_name in candidate_models:
        try:
            print(f"正在嘗試 Gemini 模型: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            # 只有當是 404 (Not Found) 時才繼續嘗試下一個，其他錯誤印出來
            print(f"模型 {model_name} 失敗: {e}")
            continue
            
    return "抱歉，目前所有 Gemini 模型都暫時無法連線，請稍後再試。"

# ========== 核心邏輯 ==========

def build_med_prompt(user_query: str) -> str:
    context = build_context_from_kb(user_query)
    return f"""你是一位運動醫學專家。
資料庫：
{context}

使用者問題：{user_query}

請根據資料庫，提供詳盡、專業的醫學與生理機制分析 (不需要口語化，請專注於專業與準確度)。
"""

def build_smart_synthesis_prompt(user_query: str, med_answer: str) -> str:
    return f"""你是一位專業但善於溝通的運動教練。
我們收到了一份來自醫學 AI 的專業分析報告，請你協助回覆使用者。

【使用者問題】
{user_query}

【醫學 AI 的專業分析】
{med_answer}

【你的任務】
請綜合以上資訊回答使用者。
**請根據使用者的問題語氣，自動決定回答風格：**
1. 如果使用者問得很專業，請保持**專業、學術**的風格。
2. 如果使用者問得很白話，請用**親切、易懂**的口語解釋。

無論哪種風格，都必須包含具體的訓練建議。
請用繁體中文回答。
"""

def answer_user_message_auto(user_query: str) -> str:
    med_answer = call_medgamma(build_med_prompt(user_query))
    final_prompt = build_smart_synthesis_prompt(user_query, med_answer)
    final_answer = call_gemini(final_prompt)
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
            answer = answer_user_message_auto(user_text)

            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=answer[:2000])
            )

    return "OK"
