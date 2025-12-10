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

# ========== 終極除錯邏輯 ==========
def test_gemini_connection(text):
    if not GEMINI_API_KEY:
        return "❌ 錯誤：Render 環境變數找不到 GEMINI_API_KEY"
    
    # 列印出版本號，確認 Render 有沒有騙我們
    try:
        import importlib.metadata
        version = importlib.metadata.version("google-generativeai")
    except:
        version = "無法取得版本"

    genai.configure(api_key=GEMINI_API_KEY)
    
    report = f"🔍 SDK版本: {version}\n"
    
    # 測試 1: Flash
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"請複誦：{text}")
        return f"✅ Flash 成功！\n回應：{response.text}"
    except Exception as e1:
        report += f"⚠️ Flash 失敗: {str(e1)}\n\n"
        
    # 測試 2: Pro (備用)
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"請複誦：{text}")
        return f"✅ Pro 成功！\n回應：{response.text}"
    except Exception as e2:
        report += f"❌ Pro 也失敗: {str(e2)}\n"
        
    report += "\n💡 診斷建議：\n如果看到 404，代表套件版本太舊或 Key 沒開權限。\n如果看到 403，代表 Key 錯誤。"
    return report

# ========== LINE Webhook ==========
@app.get("/")
async def root():
    return {"message": "Debug Mode Running"}

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
            result = test_gemini_connection(event.message.text)
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=result)
            )

    return "OK"
