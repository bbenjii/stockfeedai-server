"""Bounded catalog recovery; importing this module performs no network or database work."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import re


def normalize_symbol(value):
    if not isinstance(value, str):
        return None
    symbol = value.upper().strip()
    return symbol if re.fullmatch(r'\^?[A-Z0-9][A-Z0-9.\^=\-]{0,14}', symbol) else None


def repair_catalog(db, lookup, *, symbols=None, limit=25, workers=2, dry_run=False,
                   progress=None):
    if limit < 1 or not 1 <= workers <= 8:
        raise ValueError('Limit must be positive; workers must be between 1 and 8')
    if symbols is None:
        rows = db.articles.aggregate([
            {'$unwind': '$tickers'},
            {'$group': {'_id': '$tickers', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1, '_id': 1}},
        ])
        symbols = [r['_id'] for r in rows]
    candidates = list(dict.fromkeys(s for value in symbols if (s := normalize_symbol(value))))
    existing = set(db.stocks.distinct('symbol', {'symbol': {'$in': candidates}}))
    now = datetime.now(timezone.utc)
    waiting = set(db.untracked_symbols.distinct('symbol', {
        'symbol': {'$in': candidates}, 'retry_after': {'$gt': now},
    }))
    missing = [s for s in candidates if s not in existing]
    eligible = [s for s in missing if s not in waiting]
    result = {'candidates': len(missing), 'eligible': len(eligible),
              'attempted': 0, 'restored': 0, 'failed': 0,
              'deferred': len(missing) - len(eligible), 'dry_run': dry_run, 'halted': False}
    if dry_run:
        return result

    def fetch(symbol):
        try:
            value = lookup(symbol)
            if hasattr(value, 'model_dump'):
                value = value.model_dump()
            if (isinstance(value, dict) and value.get('symbol', '').upper() == symbol
                    and (value.get('name') or '').strip()):
                return symbol, value, None
            return symbol, None, 'temporary_failure'
        except Exception as exc:
            status = 'not_found' if getattr(exc, 'status_code', None) == 404 else 'temporary_failure'
            return symbol, None, status

    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        selected = eligible[:limit]
        for offset in range(0, len(selected), workers):
            for symbol, stock, failure in pool.map(fetch, selected[offset:offset + workers]):
                result['attempted'] += 1
                if stock:
                    doc = {key: stock.get(key) for key in ['symbol', 'name', 'currency']}
                    doc['created_at'] = datetime.now(timezone.utc)
                    db.stocks.update_one({'symbol': symbol}, {'$setOnInsert': doc}, upsert=True)
                    # A valid stock must be saved successfully before clearing its failure marker.
                    db.untracked_symbols.delete_one({'symbol': symbol})
                    result['restored'] += 1
                    consecutive_failures = 0
                else:
                    result['failed'] += 1
                    consecutive_failures = consecutive_failures + 1 if failure == 'temporary_failure' else 0
                    db.untracked_symbols.update_one({'symbol': symbol}, {
                        '$set': {'status': failure, 'reason': 'Metadata lookup unavailable',
                                 'last_attempt_at': now,
                                 'retry_after': now + timedelta(hours=168 if failure == 'not_found' else 6)},
                        '$inc': {'attempts': 1}, '$setOnInsert': {'created_at': now},
                    }, upsert=True)
                    if consecutive_failures >= 3:
                        result['halted'] = True
            if progress:
                progress(dict(result))
            if result['halted']:
                break
    return result
