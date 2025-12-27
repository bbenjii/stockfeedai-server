import os
from pymongo import MongoClient, UpdateOne, DESCENDING, ASCENDING
from typing import Dict, List
from models import Article


class ArticleService:
    def __init__(self, db=None):
        if db is None:
            uri = os.getenv("MONGO_URI")
            db_client = MongoClient(uri)
            db = db_client["dev"]
        
        
        self.collection = db.articles

    def get_articles(
            self,
            filter: Dict | None = None,
            sorting: Dict | None = None,
            limit: int | None = None,
            projection: Dict | None = None
    ) -> List[Article]:
        filter = filter or {}
        sorting = sorting or {"created_at": DESCENDING, "publish_date": DESCENDING}
        projection = projection or {}
        projection["embedding"] = False
        limit = limit or 10

        sort_list = [(k, v) for k, v in sorting.items()]

        cursor = (
            self.collection.find(filter=filter, projection=projection)
            .sort(sort_list)
        )

        if limit is not None:
            cursor = cursor.limit(limit)
        
        docs = list(cursor)
        docs = [Article(**doc).model_dump() for doc in docs]
        
        return docs