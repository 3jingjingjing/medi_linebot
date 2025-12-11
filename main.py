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

# ========== 1. 環境變數檢查 ==========

channel_secret = os.getenv("ChannelSecret", None)
channel_access_token = os.getenv("ChannelAccessToken", None)

# RunPod 設定
# 這裡會讀取你在 Render 設定的網址
RUNPOD_API_BASE = os.getenv("RUNPOD_API_BASE") 
# 讀取你在 RunPod 啟動指令設定的密碼 (medgemma-secret-key)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "medgemma-secret-key") 
# 你的模型名稱 (必須跟 RunPod 啟動指令的一模一樣)
RUNPOD_MODEL_NAME = os.getenv("RUNPOD_MODEL_NAME", "google/medgemma-27b-text-it")

if channel_secret is None:
    print("Error: ChannelSecret not set.")
    sys.exit(1)
if channel_access_token is None:
    print("Error: ChannelAccessToken not set.")
    sys.exit(1)
if RUNPOD_API_BASE is None:
    print("Warning: RUNPOD_API_BASE not set. Bot will not reply.")

# ========== 2. 初始化 LINE Bot ==========

app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# ========== 3. 知識庫 (RAG 資料) ==========

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

# ========== 4. 呼叫 RunPod (vLLM OpenAI 格式) ==========

async def call_runpod_medgemma(user_query: str) -> str:
    if not RUNPOD_API_BASE:
        return "系統錯誤：Render 未設定 RunPod 網址"

    # 準備 Prompt (醫學+教練角色)
    kb_context = build_context_from_kb(user_query)
    system_prompt = f"""你是一位專業的運動醫學專家，同時也是一位親切的教練。
請根據以下基因資料庫，回答使用者的問題。

【基因資料庫】
{kb_context}

回答原則：
1. 先從醫學角度分析生理機轉 (MedGemma 專長)。
2. 再給出白話、具體的訓練建議 (教練口吻)。
3. 請務必使用繁體中文。
"""

    # 組合正確的 API 網址
    # 你的網址是 https://...runpod.net，vLLM 需要加上 /v1/chat/completions
    # 這裡做個防呆，避免網址重複疊加
    base_url = RUNPOD_API_BASE.rstrip("/")
    if "/v1" not in base_url:
        url = f"{base_url}/v1/chat/completions"
    else:
        url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": RUNPOD_MODEL_NAME, # 例如: google/medgemma-27b-text-it
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    async with httpx.AsyncClient() as client:
        try:
            # MedGemma 27B 思考需要時間，設定 120 秒超時
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
            
            # 錯誤處理
            if resp.status_code != 200:
                return f"RunPod 連線錯誤 (Code {resp.status_code}): {resp.text}"
            
            # 解析回傳
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except httpx.ConnectError:
            return "無法連線到 RunPod，請確認 RunPod 伺服器是否開著？"
        except httpx.ReadTimeout:
            return "MedGemma 思考太久了 (Timeout)，請再試一次。"
        except Exception as e:
            return f"發生未預期的錯誤: {str(e)}"

# ========== 5. Webhook 入口 ==========

@app.get("/")
async def root():
    return {"message": "MedLineBot is running and connected to RunPod!"}

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
            
            # 顯示「處理中」的暫時回應 (可選，避免 User 以為當機)
            # await line_bot_api.push_message(event.source.user_id, TextSendMessage(text="MedGemma 正在思考中..."))
            
            # 呼叫 RunPod 取得答案
            answer = await call_runpod_medgemma(user_text)

            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=answer[:2000])
            )

    return "OK"
