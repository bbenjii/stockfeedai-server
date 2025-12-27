import os
from dotenv import load_dotenv
import certifi
load_dotenv()

from pymongo import MongoClient

from .article_service import ArticleService
from .marketdata_service import YahooStockMarket
uri = os.getenv("MONGO_URI")
db_client = MongoClient(uri, tlsCAFile=certifi.where())
db = db_client["dev"]

article_service = ArticleService(db)
marketdata_service = YahooStockMarket(db=db)

