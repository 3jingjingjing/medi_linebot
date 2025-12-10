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

# RunPod 設定 (從環境變數讀取)
# 格式範例: https://abc-123-8000.proxy.runpod.net/v1
RUNPOD_API_BASE = os.getenv("RUNPOD_API_BASE") 
# 如果你在 RunPod 有設 API Key 就填，沒有就留空
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "EMPTY") 
# 你在 RunPod 上跑的模型名稱 (要跟 RunPod 環境變數 MODEL_NAME 一樣)
RUNPOD_MODEL_NAME = os.getenv("RUNPOD_MODEL_NAME", "google/gemma-2-9b-it")

if channel_secret is None:
    print("Error: ChannelSecret not set.")
    sys.exit(1)
if channel_access_token is None:
    print("Error: ChannelAccessToken not set.")
    sys.exit(1)
if RUNPOD_API_BASE is None:
    print("Warning: RUNPOD_API_BASE not set. Bot will fail to reply.")

# ========== 初始化 ==========

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
    if not hits:
        hits = GENE_KB
    blocks = []
    for it in hits:
        blocks.append(f"基因：{it['gene']}\n特徵：{it['condition']}\n風險：{it['risk']}\n建議：{it['advice']}")
    return "\n\n".join(blocks)

# ========== 核心：呼叫 RunPod (OpenAI 格式) ==========

async def call_runpod_medgemma(system_prompt: str, user_prompt: str) -> str:
    """
    連線到 RunPod vLLM (相容 OpenAI API 格式)
    """
    if not RUNPOD_API_BASE:
        return "系統錯誤：未設定 RunPod 網址"

    # vLLM 的 Chat Completions 端點
    url = f"{RUNPOD_API_BASE}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": RUNPOD_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    async with httpx.AsyncClient() as client:
        try:
            # 設定 timeout 長一點，因為 RunPod 有時喚醒需要時間
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
            
            if resp.status_code != 200:
                return f"RunPod 錯誤 ({resp.status_code}): {resp.text}"
            
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except httpx.ConnectError:
            return "無法連線到 RunPod，請確認 Pod 是否已啟動。"
        except Exception as e:
            return f"連線發生例外: {str(e)}"

# ========== 主流程 ==========

async def process_message(user_query: str) -> str:
    # 1. 準備 RAG 資料
    kb_context = build_context_from_kb(user_query)
    
    # 2. 設定 Prompt (純文字模式，針對 MedGemma 優化)
    system_prompt = f"""
    你是一位專業的運動醫學專家，同時也是一位親切的教練。
    請根據以下使用者的基因資料庫，回答使用者的問題。
    
    【基因資料庫】
    {kb_context}
    
    回答原則：
    1. 先從醫學角度分析生理機轉。
    2. 再給出白話、具體的訓練建議。
    3. 請使用繁體中文。
    """
    
    # 3. 呼叫 RunPod
    answer = await call_runpod_medgemma(system_prompt, user_query)
    return answer

# ========== Webhook ==========

@app.get("/")
async def root():
    return {"message": "LineBot connected to RunPod is running!"}

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
            
            # 傳送處理中訊息 (因為 RunPod 算比較久，防止使用者以為壞掉)
            # (選擇性功能，這裡先直接回覆結果)
            
            answer = await process_message(user_text)

            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=answer[:2000])
            )

    return "OK"
