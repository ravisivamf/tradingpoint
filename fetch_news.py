import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BULLISH = re.compile(r"\b(?:growth|beats?|upgrad(?:e|es|ed)|surg(?:e|es|ed)|soar(?:s|ed)?|rall(?:y|ies|ied)|"
                     r"gains?|profit(?:s|able)?|strong(?:er)?|record|raises?|outperform(?:s|ed)?|jumps?)\b")
BEARISH = re.compile(r"\b(?:drops?|dropped|miss(?:es|ed)?|downgrad(?:e|es|ed)|plunge(?:s|d)?|loss(?:es)?|weak(?:er)?|"
                     r"crash(?:es|ed)?|slump(?:s|ed)?|risks?|falls?|fell|slash(?:es|ed)?|lawsuit|probe|layoffs?)\b")
MARGIN = 2   # net headline difference required before sentiment is called bullish/bearish


def get_news_detail(ticker="CRM", max_items=10, timeout=8):
    """Returns dict(coefficient, bull, bear, n, ok). coefficient: 1.2 bullish / 0.8 bearish / 1.0 neutral."""
    neutral = {"coefficient": 1.0, "bull": 0, "bear": 0, "n": 0, "ok": False}
    try:
        base = re.sub(r"(\..*|-USD|=X|=F)$", "", ticker.strip().upper().lstrip("^"))   # RELIANCE.NS -> RELIANCE, BTC-USD -> BTC
        q = urllib.parse.quote(f"{base} stock when:7d", safe="+:")
        url = f"https://news.google.com/rss/search?q={q.replace('%20', '+')}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
        titles = [(i.findtext("title") or "").lower() for i in root.findall(".//item")][:max_items]
        titles = [t for t in titles if t]
        if not titles:
            return {**neutral, "ok": True}
        bull = sum(bool(BULLISH.search(t)) for t in titles)     # one vote per headline
        bear = sum(bool(BEARISH.search(t)) for t in titles)
        coef = 1.2 if bull - bear >= MARGIN else 0.8 if bear - bull >= MARGIN else 1.0
        return {"coefficient": coef, "bull": bull, "bear": bear, "n": len(titles), "ok": True}
    except Exception:
        return neutral


def get_market_news_sentiment(ticker="CRM"):
    return get_news_detail(ticker)["coefficient"]
