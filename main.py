# -*- coding: utf-8 -*-

import os
import sys
import google.generativeai as genai
import aiohttp
from fastapi import FastAPI, Request, HTTPException
from linebot import AsyncLineBotApi, WebhookParser
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ========== 環境變數 ==========
channel_secret = os.getenv("ChannelSecret", None)
channel_access_token = os.getenv("ChannelAccessToken", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ========== 初始化 ==========
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# ========== 查錯邏輯 ==========
def test_gemini_connection(text):
    if not GEMINI_API_KEY:
        return "❌ 錯誤：Render 環境變數找不到 GEMINI_API_KEY"
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 測試順序：先測 Flash，再測 Pro
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"請複誦：{text}")
        return f"✅ 成功連線 (Flash)！回應：{response.text}"
    except Exception as e1:
        error_log = f"⚠️ Flash 連線失敗: {str(e1)}\n"
        
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(f"請複誦：{text}")
            return f"✅ 成功連線 (Pro)！回應：{response.text}"
        except Exception as e2:
            return f"❌ 全部失敗。\n錯誤詳情：{str(e1)}"

# ========== LINE Webhook ==========
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
            # 直接呼叫查錯函式
            result = test_gemini_connection(event.message.text)
            
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=result)
            )

    return "OK"
