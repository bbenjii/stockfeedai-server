from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

from services import article_service
from services.most_mentioned import companies

router = APIRouter()


@router.get("/events")
def recent_events():
    """One canonical event per cluster with publisher coverage in the last 30 days."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    rows = article_service.collection.aggregate([
        {"$match": {"cluster_id": {"$type": "string"}, "url": {"$regex": "^https?://"}}},
        {"$project": {"cluster_id": 1, "url": 1, "title": 1, "published": {"$convert": {
            "input": "$publish_date", "to": "date", "onError": None, "onNull": None}}}},
        {"$match": {"published": {"$gte": start, "$lte": end}}},
        {"$sort": {"published": -1, "_id": 1}},
        {"$group": {"_id": "$cluster_id", "published": {"$first": "$published"}, "headline": {"$first": "$title"}, "urls": {"$push": "$url"}}},
        {"$lookup": {"from": "catalyst_clusters", "localField": "_id", "foreignField": "cluster_id", "as": "cluster"}},
        {"$unwind": "$cluster"},
        {"$project": {"_id": 0, "id": "$_id", "version": "$cluster.cluster_version", "published": {"$dateToString": {"date": "$published"}}, "headline": 1,
                      "title": "$cluster.canonical_title", "summary": "$cluster.cluster_summary_short",
                      "tickers": "$cluster.tickers", "urls": {"$slice": ["$urls", 2]}}},
        {"$sort": {"published": -1, "id": 1}},
    ], maxTimeMS=18000)
    return {"start": start, "end": end, "events": list(rows), "companies": list(companies.values())}
