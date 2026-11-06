"""Market data for the hybrid-strategy replication.

OHLCV bars come only from the QuantX Dataloader REST API
(``POST /fetch-ohlcvs``). The NEFIN/FEA-USP monthly risk-factor series, which
the paper names explicitly as its factor and risk-free-rate source
(Section 3.4), is not OHLCV data and is not served by Massive; it is fetched
directly from the NEFIN website, mirroring the public CSV export it
publishes.
"""

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15  # tickers per request; batches are split by ticker, never by date
MAX_PARALLEL = 4  # concurrent requests
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore, never committed


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]  # the Massive key goes to fetch-ohlcvs only
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(
            method, API_ROOT + route, json=body, headers=headers, timeout=900
        )
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2**attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    """The wide table of one answer: inline rows, or one CSV export when stored."""
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    """Wide columns to rows of timestamp, date, ticker, open..volume."""
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            # Inline answers prefix with the raw ticker or ``alias``
            # (``AAPL_open``); the CSV export of a stored result instead
            # lower-cases the ticker verbatim, keeping any dot
            # (``cebr3.sa_open``), which differs from ``alias``
            # (``cebr3_sa``) whenever the ticker itself contains a dot.
            candidates = (
                f"{ticker}_{field}", f"{alias}_{field}",
                f"{ticker.lower()}_{field}", field,
            )
            for name in candidates:
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataloaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict], dict]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    notes = {
        "result_id": answer["result"].get("result_id") or answer.get("result_id"),
        "meta": answer.get("meta"),
    }
    return bars, details, notes


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Bars for ``tickers`` over the full inclusive window, and load metadata.

    Returns rows of timestamp (UTC), date, ticker, open, high, low, close,
    volume, sorted by ticker and time, plus metadata with every ticker's
    ``ticker_details`` entry. Tickers that failed are listed under ``failed``;
    the load fails only when no ticker loads.
    """
    request = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
            enumerate(batches),
        ))
    frames = [bars for bars, _, _ in results if not bars.empty]
    details = [detail for _, batch_details, _ in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataloaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request,
        "ticker_details": details,
        "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {
            d["ticker"]: d["data_source"]
            for d in details
            if d.get("data_source") not in (None, "massive")
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


def load_close_price_panel(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    ticker_suffix: str = ".SA",
    identity_date: str | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Wide dividend-adjusted close-price panel, one column per ticker.

    ``tickers`` are given without the exchange suffix (e.g. ``ITSA3``);
    ``ticker_suffix`` is appended for the Dataloader request and stripped back
    off the returned column names.
    """
    full_tickers = [t + ticker_suffix for t in tickers]
    bars, meta = load_ohlcv(
        full_tickers, start_date, end_date, "1d",
        adjust_dividends=True, identity_date=identity_date, use_cache=use_cache,
    )
    panel = bars.pivot(index="date", columns="ticker", values="close")
    panel.columns = [c[: -len(ticker_suffix)] if c.endswith(ticker_suffix) else c
                     for c in panel.columns]
    panel = panel.sort_index()
    return panel, meta


def load_nefin_factors(url: str, use_cache: bool = True) -> pd.DataFrame:
    """Monthly Brazilian risk factors and risk-free rate from NEFIN/FEA-USP.

    Columns: ``Rm_minus_Rf``, ``SMB``, ``HML``, ``WML``, ``IML``, ``Risk_Free``
    (monthly simple returns), indexed by a month-end ``PeriodIndex`` timestamp
    built from NEFIN's ``year``/``month`` columns. This is the market-factor
    and risk-free-rate source cited in Section 3.4.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "nefin_factors_monthly.csv"
    if use_cache and cache_path.exists():
        raw = pd.read_csv(cache_path)
    else:
        response = requests.get(url, timeout=60)
        if response.status_code != 200 or not response.content:
            raise DataloaderError(f"NEFIN factors request failed: HTTP {response.status_code}")
        raw = pd.read_csv(io.BytesIO(response.content))
        if use_cache:
            cache_path.write_bytes(response.content)
    required = {"year", "month", "Rm_minus_Rf", "SMB", "HML", "WML", "Risk_Free"}
    missing = required - set(raw.columns)
    if missing:
        raise DataloaderError(f"NEFIN factors file missing columns: {missing}")
    raw["date"] = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=1))
    raw = raw.set_index("date").sort_index()
    return raw[["Rm_minus_Rf", "SMB", "HML", "WML", "IML", "Risk_Free"]]


def load_experiment_universe(
    tickers: list[str],
    ibov_ticker: str,
    start_date: str,
    end_date: str,
    ibov_start_date: str,
    nefin_url: str,
    *,
    ticker_suffix: str = ".SA",
    identity_date: str | None = None,
    use_cache: bool = True,
) -> dict:
    """One shared load of the 13-asset panel, the Ibovespa proxy and NEFIN
    factors, used identically by every experiment (Sections 3.1 and 3.4)."""
    prices, price_meta = load_close_price_panel(
        tickers, start_date, end_date,
        ticker_suffix=ticker_suffix, identity_date=identity_date, use_cache=use_cache,
    )
    ibov_prices, ibov_meta = load_close_price_panel(
        [ibov_ticker], ibov_start_date, end_date,
        ticker_suffix=ticker_suffix, identity_date=identity_date, use_cache=use_cache,
    )
    nefin = load_nefin_factors(nefin_url, use_cache=use_cache)
    nefin = nefin[(nefin.index >= start_date) & (nefin.index <= end_date)]
    return {
        "prices": prices,
        "price_meta": price_meta,
        "ibov_prices": ibov_prices,
        "ibov_meta": ibov_meta,
        "nefin": nefin,
    }
