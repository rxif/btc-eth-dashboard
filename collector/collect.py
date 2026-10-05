import pandas as pd
import requests
import argparse
import time
from datetime import datetime, timezone
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from pathlib import Path
import logging 

BASE_URL = "https://www.deribit.com/api/v2/public"
CURRENCY = ["BTC", "ETH"]
INTERVAL_SEC = 600
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

MONTHS = {
    "JAN" : 1, "FEB" : 2, "MAR" : 3, "APR" : 4, "MAY" : 5, "JUN" : 6, 
    "JUL" : 7, "AUG" : 8, "SEP" : 9, "OCT" : 10, "NOV" : 11, "DEC" : 12
    }


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def fetch_book_summary(session: requests.Session, currency: str) -> list[dict]:
    r = session.get(
        f"{BASE_URL}/get_book_summary_by_currency", 
        params={"currency": currency, "kind": "option"},
        timeout=10,
    )
    r.raise_for_status()
    playload=r.json()
    if "result" not in playload:
        raise requests.RequestException(f"Wrong answer : {playload}")
    return playload["result"]

def parse_instrument(name: str) -> tuple[datetime, float, str]:
    _, date_str, strike_str, option_type = name.split("-")
    day = int(date_str[:-5])
    month = MONTHS[date_str[-5:-2]]
    year = 2000 +int(date_str[-2:])
    expiry = datetime(year, month, day, 8, 0, tzinfo=timezone.utc)
    strike = float(strike_str)
    return expiry, strike, option_type

def to_dataframe(rows: list[dict], currency: str, snapshot_ts: pd.Timestamp) -> pd.DataFrame:

    record = []
    for row in rows:
        try:
            expiry, strike, option_type = parse_instrument(row["instrument_name"])
        except (KeyError, ValueError) : 
            logging.warning("Instrument ignoré : %s", row.get("instrument_name"))
            continue
        
        record.append({
            "snapshot_ts": snapshot_ts,
            "currency": currency,
            "instrument_name": row.get("instrument_name"),
            "expiry": expiry,
            "strike": strike,
            "option_type": option_type,
            "bid_ask": row.get("bid_ask"),
            "ask_price": row.get("ask_price"),
            "mark_price": row.get("mark_price"),  
            "mark_iv": row.get("mark_iv"),       
            "underlying_price": row.get("underlying_price"),  
            "open_interest": row.get("open_interest"),
            "volume": row.get("volume"),
        })
    
    return pd.DataFrame.from_records(record)

def save_snapshot(df: pd.DataFrame, currency: str, snapshot_ts: pd.Timestamp) -> Path:
        
    folder = DATA_DIR / f"currency={currency}" / f"date={snapshot_ts:%Y-%m-%d}"
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"snapshot_{snapshot_ts:%H%M%S}.parquet"
    tmp=final.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(final)
        
    return final 

def run_once(session: requests.Session) -> None:
    snapshot_ts = pd.Timestamp.now(tz="UTC").floor("s")
    for currency in CURRENCY:
        try:
            rows = fetch_book_summary(session, currency)
            df = to_dataframe(rows, currency, snapshot_ts)
            if df.empty:
                logging.warning("%s : snapshot empty", currency)
                continue 
            path = save_snapshot(df, currency, snapshot_ts)
            logging.info("%s : %d option -> %s", currency, len(df), path.name)
        except Exception:
            logging.exception("%s : fail of the snapshot", currency)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="one snapshot then break")
    args = parser.parse_args()
 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    with requests.Session() as session:
        while True:
            started = time.monotonic()
            run_once(session)
            if args.once:
                break
            time.sleep(max(0.0, INTERVAL_SEC - (time.monotonic() - started)))
 
 
if __name__ == "__main__":
    main()



