## Installation

Make sure python3.13 is installed on your machine.

### Mac OS
if python3.13 is not installed, run the following command:
```bash
brew install python@3.13
```

then
``` bash
alias python="python3.13"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### Windows
```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
### Most-mentioned S&P 500 stocks

`GET /stocks/most-mentioned?sector=Technology&hours=24&limit=10` ranks distinct
article URLs per ticker by collection time (`created_at`). The window ends at
the returned UTC `as_of` minute. Omit sector for the whole index. Ties use the
latest mention, then ticker. The company’s GICS sector comes from
`services/sp500.csv`, a 2026-11-13 snapshot of Symbol, Security and GICS Sector
from https://github.com/datasets/s-and-p-500-companies/blob/main/data/constituents.csv.

## Filter articles by publisher

`GET /articles?tickers=NVDA&source=Seeking%20Alpha&limit=10` returns the latest matching articles from that publisher. Publisher names match exactly, ignoring case and surrounding whitespace. Omit `source` for all publishers.

Add `include_sources=true` to receive a `sources` array alongside `articles`. It lists named publishers matching the other filters before the publisher filter and result limit are applied.
