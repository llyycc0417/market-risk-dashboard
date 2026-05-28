import re
import time
import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

RSS_URLS = [
    "https://www.yna.co.kr/rss/international.xml"
]

KEYWORD_WEIGHTS = {
    "war": 5,
    "missile": 5,
    "airstrike": 5,
    "attack": 4,
    "military": 4,
    "iran": 4,
    "israel": 4,
    "sanctions": 3,
    "nuclear": 4,
    "oil": 3,
    "hormuz": 5,
    "conflict": 4,
    "retaliation": 4,
    "ceasefire": -2,
    "talks": -1,
    "deal": -1,

    "전쟁": 5,
    "미사일": 5,
    "공습": 5,
    "공격": 4,
    "군사": 4,
    "이란": 4,
    "이스라엘": 4,
    "제재": 3,
    "핵": 4,
    "원유": 3,
    "호르무즈": 5,
    "충돌": 4,
    "보복": 4,
    "휴전": -2,
    "협상": -1,
    "합의": -1,
}

def fetch_article_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        return text[:3000]
    except Exception:
        return ""

def score_news(title: str, summary: str, body: str):
    text = f"{title} {summary} {body}".lower()
    score = 0
    matched = []

    for kw, weight in KEYWORD_WEIGHTS.items():
        count = len(re.findall(re.escape(kw.lower()), text))
        if count > 0:
            score += count * weight
            matched.append(f"{kw}:{count}")

    if score >= 12:
        level = "HIGH"
    elif score >= 6:
        level = "MEDIUM"
    else:
        level = "LOW"

    impacts = []
    if any(k in text for k in ["oil", "원유", "hormuz", "호르무즈"]):
        impacts.append("WTI volatility ↑")
    if any(k in text for k in ["war", "missile", "attack", "전쟁", "미사일", "공격"]):
        impacts.append("VIX ↑")
    if score >= 10:
        impacts.append("S&P500 risk ↑")

    return score, level, ", ".join(matched), " / ".join(impacts)

def update_news(output_csv="today_geopolitical_news.csv", top_n=10):
    rows = []

    for rss_url in RSS_URLS:
        feed = feedparser.parse(rss_url)

        for entry in feed.entries:
            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")
            summary = getattr(entry, "summary", "")

            published = getattr(entry, "published", "")
            body = fetch_article_text(link)

            score, level, matched, impact = score_news(title, summary, body)

            rows.append({
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "published": published,
                "title": title,
                "url": link,
                "risk_score": score,
                "risk_level": level,
                "matched_keywords": matched,
                "expected_impact": impact,
                "summary": BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)[:300],
            })

            time.sleep(0.3)

    df = pd.DataFrame(rows)

    if df.empty:
        print("수집된 뉴스가 없습니다.")
        return

    df = df.sort_values("risk_score", ascending=False).head(top_n)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {output_csv}")
    print(df[["title", "risk_score", "risk_level"]])

if __name__ == "__main__":
    update_news()