import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from services import article_service

router = APIRouter()
with Path(__file__).with_name("sp500.csv").open(encoding="utf-8") as catalog:
    companies = {row["ticker"]: row for row in csv.DictReader(catalog)}
sectors = sorted({company["sector"] for company in companies.values()})


@lru_cache(maxsize=16)
def ranking(hours, minute):
    end = datetime.fromtimestamp(minute * 60, timezone.utc)
    rows = article_service.collection.aggregate([
        {"$match": {"created_at": {"$gte": end - timedelta(hours=hours), "$lt": end},
                    "url": {"$type": "string", "$ne": ""}}},
        {"$unwind": "$tickers"},
        {"$match": {"tickers": {"$type": "string"}}},
        {"$project": {"url": 1, "created_at": 1, "ticker": {"$replaceAll": {
            "input": {"$toUpper": {"$trim": {"input": "$tickers"}}}, "find": "-", "replacement": "."}}}},
        {"$match": {"ticker": {"$in": list(companies)}}},
        {"$group": {"_id": {"ticker": "$ticker", "url": "$url"}, "latest": {"$max": "$created_at"}}},
        {"$group": {"_id": "$_id.ticker", "article_count": {"$sum": 1}, "latest": {"$max": "$latest"}}},
        {"$sort": {"article_count": -1, "latest": -1, "_id": 1}},
    ], maxTimeMS=12000)
    return [{**companies[row["_id"]], "article_count": row["article_count"]} for row in rows]


@router.get("/stocks/most-mentioned")
def most_mentioned(sector: str | None = None, hours: int = Query(24, ge=1, le=168),
                   limit: int = Query(10, ge=1, le=100)):
    if sector:
        aliases = {s.casefold(): s for s in sectors}
        aliases["technology"] = "Information Technology"
        sector = aliases.get(sector.strip().casefold())
        if not sector:
            raise HTTPException(422, "Unknown company sector")
    minute = int(datetime.now(timezone.utc).timestamp()) // 60
    stocks = [s for s in ranking(hours, minute) if not sector or s["sector"] == sector]
    return {"stocks": stocks[:limit], "sectors": sectors, "hours": hours,
            "as_of": datetime.fromtimestamp(minute * 60, timezone.utc)}
