from __future__ import annotations

import os
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf
from fastapi import HTTPException
from pydantic import BaseModel
from pymongo import InsertOne, UpdateOne
from starlette import status

from utils import function_timer, logger

logger.setLevel("INFO")

def timestamp_to_datetime(value: Optional[float]) -> Optional[str]:
    """Convert various timestamp formats to a UTC string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt_obj = value.astimezone(timezone.utc)
    elif isinstance(value, (int, float)):
        dt_obj = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        return None
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


class Ticker(BaseModel):
    symbol: str
    name: str
    currency: Optional[str] = None
    regularMarketTime: Optional[str] = None
    regularMarketPrice: Optional[float] = None
    regularMarketDayHigh: Optional[float] = None
    regularMarketDayLow: Optional[float] = None
    fiftyTwoWeekHigh: Optional[float] = None
    fiftyTwoWeekLow: Optional[float] = None


class MarketCandle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int]


class YahooStockMarket:
    """
    Wrapper around yfinance to keep a consistent interface with the previous Yahoo API client.
    """

    def __init__(self, *, db=None, auto_adjust: bool = False, ) -> None:
        self.auto_adjust = auto_adjust
        self.db = db
        self.initialize_stocks_collection()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone
    from pymongo import UpdateOne
    import os

    @function_timer
    def initialize_stocks_collection(
            self,
            max_symbols: int | None = None,
            chunk_size: int = 50,
            max_workers: int | None = 8,
    ):
        pipeline = [
            {"$unwind": "$tickers"},
            {"$group": {"_id": "$tickers", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$project": {"ticker": "$_id", "_id": 0}},
        ]
        if max_symbols:
            pipeline.append({"$limit": max_symbols})

        symbols = list(self.db.articles.aggregate(pipeline, allowDiskUse=True))
        symbols_names = [s["ticker"].upper().strip() for s in symbols if s.get("ticker")]

        if not symbols_names:
            return {"inserted_or_upserted": 0, "candidates": 0, "missing": 0}

        existing = set(self.db.stocks.distinct("symbol", {"symbol": {"$in": symbols_names}}))
        untracked = set(self.db.untracked_symbols.distinct("symbol", {"symbol": {"$in": symbols_names}}))

        # only attempt those not in stocks AND not in untracked_symbols
        missing = [s for s in symbols_names if s not in existing and s not in untracked]

        if not missing:
            return {
                "inserted_or_upserted": 0,
                "candidates": len(symbols_names),
                "missing": 0,
                "skipped_untracked": len(untracked),
            }

        now = datetime.now(timezone.utc)

        if max_workers is None:
            max_workers = min(16, (os.cpu_count() or 4) * 2)

        operations: list[UpdateOne] = []
        untracked_ops: list[UpdateOne] = []
        upserted_total = 0
        newly_untracked_total = 0

        def flush_stock_ops() -> int:
            nonlocal operations
            if not operations:
                return 0
            res = self.db.stocks.bulk_write(operations, ordered=False)
            operations = []
            return res.upserted_count or 0

        def flush_untracked_ops() -> int:
            nonlocal untracked_ops
            if not untracked_ops:
                return 0
            res = self.db.untracked_symbols.bulk_write(untracked_ops, ordered=False)
            untracked_ops = []
            return res.upserted_count or 0

        def maybe_flush_untracked():
            nonlocal newly_untracked_total
            if len(untracked_ops) >= chunk_size:
                newly_untracked_total += flush_untracked_ops()

        def maybe_flush_stocks():
            nonlocal upserted_total
            if len(operations) >= chunk_size:
                upserted_total += flush_stock_ops()

        def mark_untracked(symbol: str, reason: str | None = None):
            doc = {"symbol": symbol, "created_at": now}
            if reason:
                doc["reason"] = reason

            untracked_ops.append(
                UpdateOne(
                    {"symbol": symbol},
                    {"$setOnInsert": doc},
                    upsert=True,
                )
            )
            maybe_flush_untracked()

        def fetch(symbol: str):
            try:
                return self.get_stock_info(symbol=symbol)
            except Exception as exc:
                return ("__error__", symbol, str(exc))

        processed = 0
        ok = 0
        failed = 0
        log_every = 50

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch, s): s for s in missing}

            for fut in as_completed(futures):
                processed += 1
                result = fut.result()

                # exception path
                if isinstance(result, tuple) and result and result[0] == "__error__":
                    _, symbol, err = result
                    failed += 1
                    mark_untracked(symbol, reason=err)

                else:
                    t = result
                    if not t:
                        failed += 1
                        mark_untracked(futures[fut])  # symbol string
                    else:
                        ok += 1
                        operations.append(
                            UpdateOne(
                                {"symbol": t.symbol},
                                {
                                    "$setOnInsert": {
                                        "symbol": t.symbol,
                                        "name": t.name,
                                        "currency": t.currency,
                                        "created_at": now,
                                    }
                                },
                                upsert=True,
                            )
                        )
                        maybe_flush_stocks()

                # regular flushing + progress logging
                if processed % log_every == 0:
                    newly_untracked_total += flush_untracked_ops()
                    upserted_total += flush_stock_ops()
                    logger.info(
                        "Processed %s/%s (ok=%s, failed=%s)",
                        processed, len(missing), ok, failed
                    )

        # final flush
        upserted_total += flush_stock_ops()
        newly_untracked_total += flush_untracked_ops()

        return {
            "inserted_or_upserted": upserted_total,
            "newly_untracked": newly_untracked_total,
            "candidates": len(symbols_names),
            "missing_attempted": len(missing),
            "skipped_untracked": len(untracked),
            "processed": processed,
            "ok": ok,
            "failed": failed,
            "max_workers": max_workers,
        }

    def _dataframe_to_candles(self, df: pd.DataFrame) -> List[MarketCandle]:
        if df.empty:
            return []

        candles: List[MarketCandle] = []
        for index, row in df.iterrows():
            ts = timestamp_to_datetime(index.to_pydatetime() if hasattr(index, "to_pydatetime") else index)
            if ts is None:
                continue
            if pd.isna(row["Open"]) or pd.isna(row["Close"]):
                continue
            candles.append(
                MarketCandle(
                    date=ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=None if pd.isna(row["Volume"]) else int(row["Volume"]),
                )
            )
        return candles
    
    def _search_stocks_collection(self, search: str, limit: int) -> list[dict]:

        regex = {"$regex": search, "$options": "i"}

        cursor = (
            self.db.stocks.find(
                {
                    "$or": [
                        {"symbol": regex},
                        {"name": regex},
                    ]
                },
                {
                    "_id": 0,
                    "created_at": 0,
                },
            )
            .limit(limit)
        )

        return list(cursor)

    def _search_from_articles(self, search: str, limit: int, exclude_symbols: set[str] | None = None):
        exclude_symbols = exclude_symbols or set()
        candidate_limit = max(limit * 3, 30)

        pipeline = [
            {"$unwind": "$tickers"},
            {"$match": {"tickers": {"$regex": search, "$options": "i"}}} if search else {"$match": {}},
            {"$group": {"_id": "$tickers", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": candidate_limit},
            {"$project": {"ticker": "$_id", "_id": 0}},
        ]

        symbols = list(self.db.articles.aggregate(pipeline, allowDiskUse=True))
        symbols_names = [s["ticker"] for s in symbols if s.get("ticker")]

        results = []

        for s in symbols_names:
            if s in exclude_symbols:
                continue

            try:
                t = self.get_stock_info(symbol=s)
            except Exception:
                continue

            if not t:
                continue

            results.append(t.model_dump())

            # optional: persist for future fast lookups
            try:
                self.db.stocks.update_one(
                    {"symbol": t.symbol},
                    {
                        "$setOnInsert": {
                            "symbol": t.symbol,
                            "name": t.name,
                            "currency": t.currency,
                            "created_at": datetime.now(timezone.utc),
                        }
                    },
                    upsert=True,
                )
            except Exception:
                pass

            if len(results) >= limit:
                break

        return results

    def get_stock_symbols(self, search: str = "", limit: int = 10):
        # ---- A: fast path (cached stocks)
        stocks = self._search_stocks_collection(search, limit)

        return stocks[:limit]

    @staticmethod
    def _safe_fast_info_get(fast_info, key: str, default=None):
        """
        yfinance fast_info isn't always a dict and can raise KeyError internally.
        This shields your API from that.
        """
        if not fast_info:
            return default

        # dict-like
        try:
            if isinstance(fast_info, dict):
                return fast_info.get(key, default)
        except Exception:
            return default

        # yfinance FastInfo object: may raise KeyError/Exception
        try:
            return fast_info.get(key, default)  # may still raise
        except Exception:
            pass

        try:
            return fast_info[key]  # may raise
        except Exception:
            return default

    def get_stock_info(self, symbol: str = "AVGO") -> Optional[Ticker]:
        symbol = symbol.upper().strip()
        ticker = yf.Ticker(symbol)

        try:
            info: Dict = ticker.get_info() or {}
        except Exception as exc:  # remote errors
            logger.warning("Unable to fetch %s info via yfinance: %s", symbol, exc)
            return None

        fast_info = getattr(ticker, "fast_info", None)

        currency = info.get("currency") or self._safe_fast_info_get(fast_info, "currency")
        last_price = info.get("regularMarketPrice") or self._safe_fast_info_get(fast_info, "last_price")
        day_high = info.get("regularMarketDayHigh") or self._safe_fast_info_get(fast_info, "day_high")
        day_low = info.get("regularMarketDayLow") or self._safe_fast_info_get(fast_info, "day_low")
        year_high = info.get("fiftyTwoWeekHigh") or self._safe_fast_info_get(fast_info, "year_high")
        year_low = info.get("fiftyTwoWeekLow") or self._safe_fast_info_get(fast_info, "year_low")

        # If Yahoo doesn't know the symbol, info often comes back nearly empty.
        # Optional guard: require at least a name or price to accept it.
        name = info.get("shortName") or info.get("longName") or symbol
        
        if name == symbol and last_price is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticker '{symbol}' not found"
            )
        
        return Ticker(
            symbol=symbol,
            name=name,
            currency=currency,
            regularMarketTime=timestamp_to_datetime(info.get("regularMarketTime")),
            regularMarketPrice=last_price,
            regularMarketDayHigh=day_high,
            regularMarketDayLow=day_low,
            fiftyTwoWeekHigh=year_high,
            fiftyTwoWeekLow=year_low,
        )

    PeriodInterval = Literal["1m", "1h", "1d", "1w", "1mo"]

    def get_multiple_stock_df(
            self,
            ticker_symbols: list[str],
            interval: PeriodInterval = "1d",
            length: int = 10,
            end_time=None,
            prepost=False,
    ):
        df_set = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            fut_map = {pool.submit(self.get_stock_df, ticker_symbol=ticker_symbol, interval=interval, length=length,
                                   end_time=end_time, prepost=prepost): (i, ticker_symbol) for i, ticker_symbol in
                       enumerate(ticker_symbols)}
            for fut in as_completed(fut_map):
                i, ticker_symbol = fut_map[fut]
                try:
                    df = fut.result()
                    if df is not None:
                        df_set[ticker_symbols[i]] = df
                except Exception as e:
                    logger.warning("Worker failed for %s: %s", ticker_symbol, e)

        return df_set

    def get_stock_df(
            self,
            ticker_symbol: str,
            interval: PeriodInterval = "1d",
            length: int = 10,
            end_time=None,
            prepost=False,

    ):
        ticker = yf.Ticker(ticker_symbol)
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        interval_map = {
            "1m": timedelta(minutes=length),
            "1h": timedelta(hours=length),
            "1d": timedelta(days=length),
            "1w": timedelta(weeks=length),
            "1mo": timedelta(days=30 * length),
        }

        start_time = end_time - interval_map[interval]

        try:
            df = ticker.history(
                start=start_time,
                end=end_time,
                interval=interval,
                auto_adjust=self.auto_adjust,
                actions=False,
                prepost=prepost,
            )
        except Exception as exc:  # pragma: no cover - remote errors
            logger.error("Unable to fetch %s history via yfinance: %s", ticker_symbol, exc)
            return None

        return df

    def get_stock_history(
            self,
            symbol: str = "AVGO",
            days: int = 7,
            interval: PeriodInterval = "1d",
            start_time=None, end_time=None,
            prepost=False,
    ):

        ticker = yf.Ticker(symbol)
        if start_time and end_time:
            end = end_time
            start = start_time
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)

        try:
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=self.auto_adjust,
                actions=False,
                prepost=prepost,
            )
        except Exception as exc:  # pragma: no cover - remote errors
            logger.error("Unable to fetch %s history via yfinance: %s", symbol, exc)
            return None
        index_name = df.index.name
        df = df.reset_index().to_dict('records')

        for entry in df:
            entry["time"] = int(entry[index_name].timestamp())
            del entry[index_name]

        ticker_info = self.get_stock_info(symbol)
        if ticker_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticker '{symbol}' not found"
            )
        return {"candles": df,
                "ticker": ticker_info.model_dump(),
                }

    # @function_timer
    def get_intraday_history(
            self,
            symbol: str = "AVGO",
            interval: str = "1m",
            include_pre_post: bool = True,
    ) -> Optional[List[MarketCandle]]:
        ticker = yf.Ticker(symbol)
        try:
            df = ticker.history(
                period="5d",
                interval=interval,
                auto_adjust=self.auto_adjust,
                actions=False,
                prepost=include_pre_post,
            )
        except Exception as exc:
            logger.error("Unable to fetch %s intraday data via yfinance: %s", symbol, exc)
            return None

        if df.empty:
            return []

        # Filter to the most recent trading session
        normalized = df.index.normalize()
        target_day = normalized.max()
        session_df = df[normalized == target_day]
        if session_df.empty and len(normalized.unique()) > 1:
            # fallback to previous day if last day has no rows (rare)
            second_last = sorted(set(normalized))[-2]
            session_df = df[normalized == second_last]

        candles = self._dataframe_to_candles(session_df)
        length = len(candles)
        return candles

    def get_momentum(self):
        pass

    def get_most_active_symbols(self, count: int = 200, query="") -> List[Dict[str, Optional[str]]]:
        try:
            result = yf.screen(query if query else "most_actives", count=count)
        except Exception as exc:
            logger.error("Unable to fetch most active symbols via yfinance: %s", exc)
            return []

        entries = result.get("quotes") or result.get("records") or []
        symbols: List[Dict[str, Optional[str]]] = []
        for entry in entries[:count]:
            symbol = entry.get("symbol") or entry.get("ticker")
            if not symbol:
                continue
            symbols.append(
                {
                    "symbol": symbol,
                    "shortName": entry.get("shortName") or entry.get("companyName"),
                }
            )
        return symbols


def continuous_fetch(symbol: str = "NVDA", days: int = 5, delay_seconds: float = 2.0) -> None:
    """
    Simple helper to repeatedly poll yfinance for manual profiling or debugging.
    """
    stock_market = YahooStockMarket()

    num = 0
    stock_cache: Optional[List[MarketCandle]] = None
    start_time = time.perf_counter()

    while True:
        stock = stock_market.get_stock_info(symbol=symbol)
        if stock:
            num += 1
            stock_cache = stock
            # print(f"Fetch number: {num}")
            # print(f"Elapsed time: {time.perf_counter() - start_time:.4f} seconds")
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now}: {round(stock_cache.regularMarketPrice, 4)} $")
            time.sleep(delay_seconds)
        else:
            break

    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Number of fetches: {num}")
    if stock_cache:
        print(f"Number of candles: {len(stock_cache)}")


def main() -> None:
    import os
    from dotenv import load_dotenv
    import certifi
    load_dotenv()
    from pymongo import MongoClient

    uri = os.getenv("MONGO_URI")
    db_client = MongoClient(uri, tlsCAFile=certifi.where())
    db = db_client["dev"]

    symbol = "NVDA"
    stock_market = YahooStockMarket(db=db)
    stock_market.initialize_stocks_collection()
    # symbols = stock_market.get_stock_symbols("")
    # most_active = stock_market.get_most_active_symbols(count=10, query="META")
    pass
    # ticker = stock_market.get_stock_info(symbol=symbol)
    # stock_history = stock_market.get_stock_history(symbol=symbol, days=365 * 3, interval="1d") or []
    # 
    # intraday = stock_market.get_intraday_history(symbol=symbol, include_pre_post=True) or []
    # most_active = stock_market.get_most_active_symbols(count=10)
    # 
    # print(ticker)
    # print(f"Fetched {len(stock_history)} daily candles for {symbol}")
    # print(f"Fetched {len(intraday)} intraday candles for {symbol}")
    # print("Most active symbols:", [entry["symbol"] for entry in most_active if entry.get("symbol")])
    

if __name__ == "__main__":
    # continuous_fetch(delay_seconds=0.5)
    main()
    # print(YahooStockMarket().get_stock_info("$BRK"))
