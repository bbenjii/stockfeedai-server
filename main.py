import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services import article_service, marketdata_service
from typing import Optional, Dict, Any
from fastapi import Query
from fastapi.responses import JSONResponse
import re
from datetime import datetime, timedelta, timezone, date

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "Hello World"}

from fastapi import Query
from typing import Any, Dict
import re

@app.get("/article/{article_slug}")
async def search_article(article_slug: str):
    # Escape slug to avoid regex injection
    slug_rx = re.escape(article_slug)

    mongo_filter: Dict[str, Any] = {
        "url": {
            "$regex": slug_rx,
            "$options": "i",
        }
    }

    articles = article_service.get_articles(
        filter=mongo_filter,
        limit=1,
    )

    article = articles[0] if articles else None

    return JSONResponse(
        content={"article": article},
        status_code=200,
    )

    
    
@app.get("/articles")
async def get_articles(
    search: Optional[str] = Query(default=None),
    tickers: Optional[str] = Query(default=None),
    sectors: Optional[str] = Query(default=None),
    sentiment: Optional[str] = Query(default=None),
    hours: Optional[int] = Query(default=None, ge=1, le=24*30),
    only_with_tickers: bool = Query(default=True),
    limit: int = Query(default=20, ge=1, le=500),
):
    mongo_filter: Dict[str, Any] = {}

    # Time window 
    if hours is not None:
        since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
        mongo_filter["created_at"] = {"$gte": since_dt}

    # Free-text search
    if search and search.strip():
        q = re.escape(search.strip())
        rx = {"$regex": q, "$options": "i"}
        search_or = [
            {"title": rx},
            {"content": rx},
            {"summary": rx},
            {"keyword": rx},
            {"keywords": rx},
            {"sectors": rx},
            {"tickers": rx},
            {"authors": rx},
            {"url": rx},
        ]
        # Merge with existing $or (from time filter) using $and
        if "$or" in mongo_filter:
            mongo_filter = {"$and": [mongo_filter, {"$or": search_or}]}
        else:
            mongo_filter["$or"] = search_or

    # Tickers filter
    ticker_conditions = []

    if tickers and tickers.strip():
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if ticker_list:
            ticker_conditions.append({"$in": ticker_list})

    # Only with tickers
    if only_with_tickers:
        ticker_conditions.append({"$exists": True})
        ticker_conditions.append({"$ne": []})

    if ticker_conditions:
        mongo_filter["tickers"] = {}
        for condition in ticker_conditions:
            mongo_filter["tickers"].update(condition)

    # Sectors filter
    if sectors and sectors.strip():
        sector_list = [s.strip() for s in sectors.split(",") if s.strip()]
        if sector_list:
            mongo_filter["sectors"] = {"$in": sector_list}

    # Sentiment filter
    if sentiment and sentiment.strip():
        s = sentiment.strip().lower()
        if s in {"positive", "negative", "neutral"}:
            mongo_filter["sentiment"] = s

    
    print(mongo_filter)
    articles = article_service.get_articles(filter=mongo_filter, limit=limit)
    return JSONResponse(content={"articles": articles}, status_code=200)


@app.get("/stock/symbols")
async def get_stock_symbols(search: Optional[str] = Query(default=""),):
    
    symbols = marketdata_service.get_stock_symbols(search)
    return JSONResponse(content={"symbols": symbols}, status_code=200)
@app.get("/stock/{ticker}/history")
async def get_stock_history(ticker: str,     
                            period: Optional[str] = Query(default="1d"),
):
    def ytd_days(d: date | None = None) -> int:
        d = d or date.today()
        start_of_year = date(d.year, 1, 1)
        return (d - start_of_year).days
    
    def get_period_settings(period: str) -> Dict[str, Any]:
        period_settings_map = {
            "1d": {"days": 1, "interval": "1m"},
            "5d": {"days": 5, "interval": "1m"},
            "1mo": {"days": 30, "interval": "1h"},
            "3mo": {"days": 90, "interval": "1h"},
            "6mo": {"days": 180, "interval": "1h"},
            "ytd": {"days": ytd_days(), "interval": "1h"},
            "1y": {"days": 365, "interval": "1h"},
            "3y": {"days": 365 * 3, "interval": "1d"},
            "5y": {"days": 365 * 5, "interval": "1d"},
        }
        
        return period_settings_map.get(period, {})
    
    period_settings = get_period_settings(period)

    stock_history = marketdata_service.get_stock_history(symbol=ticker, **get_period_settings(period))
    return JSONResponse(content={"history": stock_history}, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)