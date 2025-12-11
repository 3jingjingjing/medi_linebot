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

# ========== 設定區 ==========

channel_secret = os.getenv("ChannelSecret", None)
channel_access_token = os.getenv("ChannelAccessToken", None)

# 讀取遠端 API 設定 (連線到朋友的 ngrok)
REMOTE_API_BASE = os.getenv("REMOTE_API_BASE") 
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "EMPTY") 
REMOTE_MODEL_NAME = os.getenv("REMOTE_MODEL_NAME", "google/medgemma-27b-text-it")

if channel_secret is None:
    print("Error: ChannelSecret not set.")
    sys.exit(1)
if channel_access_token is None:
    print("Error: ChannelAccessToken not set.")
    sys.exit(1)
if REMOTE_API_BASE is None:
    print("Warning: REMOTE_API_BASE not set. Bot will not reply.")

app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# ========== 知識庫 (RAG) ==========

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
    hits = [item for item in GENE_KB if item["gene"] in user_query]
    if not hits: hits = GENE_KB
    blocks = []
    for it in hits:
        blocks.append(f"基因：{it['gene']}\n特徵：{it['condition']}\n風險：{it['risk']}\n建議：{it['advice']}")
    return "\n\n".join(blocks)

# ========== 核心：呼叫遠端模型 (ngrok) ==========

async def call_remote_medgemma(user_query: str) -> str:
    if not REMOTE_API_BASE:
        return "系統錯誤：未設定遠端 API 網址 (REMOTE_API_BASE)"

    # 1. 準備 Prompt
    kb_context = build_context_from_kb(user_query)
    system_prompt = f"""你是一位專業運動醫學專家。
請根據以下基因資料庫回答問題：
{kb_context}
回答時請先分析生理機制，再給出具體訓練建議。請用繁體中文。
"""

    # 2. 準備 Payload
    headers = {
        "Authorization": f"Bearer {REMOTE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": REMOTE_MODEL_NAME, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    # 3. 自動路徑偵測 (適應 vLLM 或其他 Server)
    clean_base = REMOTE_API_BASE.rstrip("/")
    candidate_urls = [
        f"{clean_base}/v1/chat/completions",  # 標準 vLLM / OpenAI 格式
        f"{clean_base}/chat/completions",     # 某些簡易 Server 格式
    ]

    async with httpx.AsyncClient() as client:
        last_error = ""
        
        for url in candidate_urls:
            try:
                # 設定長一點的 timeout (因為家用電腦網路可能比較慢)
                resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
                
                if resp.status_code == 404:
                    continue # 路徑不對，換下一個
                
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                
                return f"遠端伺服器錯誤 ({resp.status_code}): {resp.text}"

            except httpx.ConnectError:
                return "無法連線到 ngrok。請確認朋友電腦是否開著，且 ngrok 正在執行？"
            except Exception as e:
                last_error = str(e)

        return f"連線失敗。請確認 ngrok 網址是否正確。\n最後錯誤: {last_error}"

# ========== Webhook ==========

@app.get("/")
async def root():
    return {"message": "MedLineBot connected via ngrok is running!"}

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
            answer = await call_remote_medgemma(user_text)
            await line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer[:2000]))

    return "OK"
