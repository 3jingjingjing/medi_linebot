# -*- coding: utf-8 -*-

#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.

import os
import sys
import httpx

import aiohttp
from fastapi import FastAPI, Request, HTTPException

from linebot import AsyncLineBotApi, WebhookParser
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ========== LINE Bot 設定 ==========

channel_secret = os.getenv("ChannelSecret", None)
channel_access_token = os.getenv("ChannelAccessToken", None)

if channel_secret is None:
    print("Specify ChannelSecret as environment variable.")
    sys.exit(1)
if channel_access_token is None:
    print("Specify ChannelAccessToken as environment variable.")
    sys.exit(1)

app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# ========== 你的運動基因知識庫（Python 版 RAG 資料） ==========

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
    # TODO: 在這裡繼續補上你的其他基因與說明
]


def match_gene_items(user_query: str):
    """
    超簡易版「RAG 檢索」：
    - 如果問題裡有提到某個基因名稱，就只選那些
    - 如果完全沒 match，就先全部給（之後可以升級成向量資料庫）
    """
    hits = []
    for item in GENE_KB:
        if item["gene"] in user_query:
            hits.append(item)
    if not hits:
        hits = GENE_KB
    return hits


def build_context_from_kb(user_query: str) -> str:
    """
    把上面 GENE_KB 裡挑出的條目，組成一段給 LLM 用的 context 字串。
    """
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


# ========== MedGamma（MedGemma）與 Gemini API client ==========

MEDGAMMA_ENDPOINT = os.getenv("MEDGAMMA_ENDPOINT")
MEDGAMMA_API_KEY = os.getenv("MEDGAMMA_API_KEY")

GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def call_medgamma(prompt: str) -> str:
    """
    呼叫 MedGamma（或你部署的 MedGemma 醫學模型）API。
    """
    if not MEDGAMMA_ENDPOINT:
        raise RuntimeError("MEDGAMMA_ENDPOINT not set")

    payload = {
        "input": prompt,
        # 這裡依你實際 MedGamma API 格式調整
    }
    headers = {"Content-Type": "application/json"}
    if MEDGAMMA_API_KEY:
        headers["Authorization"] = f"Bearer {MEDGAMMA_API_KEY}"

    resp = httpx.post(MEDGAMMA_ENDPOINT, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # 這行要依照實際回傳格式調整
    return data["output"]


def call_gemini(prompt: str) -> str:
    """
    呼叫 Gemini（例如 Google Generative AI）的 API，做口語化解釋。
    """
    if not GEMINI_ENDPOINT:
        raise RuntimeError("GEMINI_ENDPOINT not set")

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
        # 若你是用 Google 官方 SDK / 其他 endpoint，這裡要配合修改
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }

    resp = httpx.post(GEMINI_ENDPOINT, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # 同樣依照實際回傳格式調整
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ========== prompt 組裝邏輯 ==========

def build_med_prompt(user_query: str) -> str:
    """
    給 MedGamma 用的專業版 prompt：會帶入你的 Python 知識庫（RAG context）。
    """
    context = build_context_from_kb(user_query)
    return f"""你是一位運動醫學與運動基因專家。

【運動基因與跑步相關知識庫】
{context}

【使用者問題】
{user_query}

請你根據上述知識庫內容，從醫學與運動生理角度分析：
1. 說明該跑者在跑步／馬拉松訓練上的潛在優勢與風險。
2. 說明可能涉及的生理機轉（例如發炎反應、肌肉損傷、恢復速度等），但不要捏造沒有依據的內容。
3. 提供清楚、具體的訓練與恢復建議。

請用繁體中文回答，內容可以專業一點沒關係。
"""


def build_gemini_prompt(user_query: str, med_answer: str) -> str:
    """
    給 Gemini 的 prompt：請他把 MedGamma 的專業說明翻譯成跑者聽得懂的版本。
    """
    return f"""以下是針對一位跑者的運動基因與訓練風險，由醫學模型產生的專業說明：

【專業說明】
{med_answer}

請你把上面的內容重新整理成：
1. 一般跑者也聽得懂的白話解釋。
2. 用條列方式列出 3~5 個重點。
3. 給出具體、容易執行的訓練建議（例如配速、每週跑量、恢復時間、熱身與收操注意事項）。

請用繁體中文回答，語氣友善但務實，像是在對跑者講話。
使用者原本的問題是：「{user_query}」。
"""


def answer_user_message(user_query: str, mode: str = "friendly") -> str:
    """
    整合 MedGamma + Gemini 的主流程。

    mode:
      - "pro"：只回 MedGamma 的專業版（適合你自己或教練看）
      - "friendly"：MedGamma 做底，再交給 Gemini 口語化（給一般跑者）
    """
    med_prompt = build_med_prompt(user_query)
    med_answer = call_medgamma(med_prompt)

    if mode == "pro":
        return med_answer

    gemini_prompt = build_gemini_prompt(user_query, med_answer)
    final_answer = call_gemini(gemini_prompt)
    return final_answer


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
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessage):
            continue

        user_text = event.message.text.strip()

        # 若訊息前面加 #專業 ，就回純醫學版
        if user_text.startswith("#專業"):
            query = user_text.replace("#專業", "", 1).strip()
            answer = answer_user_message(query, mode="pro")
        else:
            # 預設：一般跑者模式 → 口語化版本
            answer = answer_user_message(user_text, mode="friendly")

        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=answer[:2000])  # 避免超過 LINE 長度限制
        )

    return "OK"
