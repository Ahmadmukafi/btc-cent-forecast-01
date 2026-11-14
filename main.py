#!/usr/bin/env python3
"""BTC 4-hour-ahead point forecaster for make.fun btc-cent-forecast-01.

Model
-----
The forecast is the checkpoint close carried forward by a damped slice of
the very recent drift:

    guess = last * exp(0.75 * log(last / price_60_min_ago))

In words: take the log-return of the last 60 minutes and apply 75% of it
to the next 4 hours.  Bitcoin's minute prices are mostly flat with rare
jumps, so the last hour's drift is the freshest signal of where the price
is headed, and the 0.75 damping shrinks it toward the persistence
baseline (guess = last close) to avoid chasing noise.

Why this shape
--------------
* Scoring is mean absolute error, so the best point forecast is the
  conditional median of the 4h-ahead price.  A damped short-horizon
  momentum term is a robust estimate of that median's drift.
* Longer windows (4h/12h/24h returns, OLS slopes, median block returns)
  were tested on the six practice cases and all did worse: extrapolating
  a fitted slope over 240 minutes amplifies noise, and long-window
  returns lag the regime the price is actually in.
* The 0.75 factor sits in the flat part of the practice loss curve
  (0.70 -> 33,267; 0.75 -> 32,283; 0.80 -> 32,451; 1.00 -> 33,491 cents
  mean absolute error over the five practice cases whose actuals are
  recoverable, versus 47,055 for the persistence baseline), so the exact
  value is not over-tuned to a handful of cases.

Interface
---------
Reads every ``*.in`` file in $MAKEFUN_INPUT_DIR (default /input) in name
order.  Each case is ``{"schema": "makefun/btc-cent-case/v1", "case": id,
"at": checkpoint, "horizon_minutes": 240, "markets": [{"id": "btc-usdc",
"prices": [4320 one-minute closes]}]}``.  Writes ``/output/<id>.out``
(or $MAKEFUN_OUTPUT_DIR) as ``{"forecasts": [{"case": id, "price": n}]}``.

Deterministic (same input -> same output, byte for byte), standard
library only, no network, no model calls, runs in milliseconds.
"""

import json
import math
import os

# Share of the last hour's log-return carried into the 4h forecast.
MOMENTUM_SHARE = 0.75
# Safety cap on the forecast move, so one bad tick can never blow the
# guess up.  Practice 1h returns stay well under 1%, so this never binds
# there; it only guards against pathological inputs.
MAX_MOVE = 0.05


def forecast(prices):
    """Return the 4h-ahead price guess for one case's close series."""
    last = prices[-1]
    ref = prices[-61] if len(prices) > 60 else prices[0]
    if last <= 0 or ref <= 0:
        return last
    drift = MOMENTUM_SHARE * math.log(last / ref)
    drift = max(-MAX_MOVE, min(MAX_MOVE, drift))
    return last * math.exp(drift)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        # The output file is named after the input file: /output/<id>.out
        # for /input/<id>.in, so the case id comes from the filename.
        case_id = name[:-3]
        path = os.path.join(in_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                case = json.load(f)
            prices = case["markets"][0]["prices"]
            price = forecast(prices)
        except Exception:
            # Never let one bad case kill the whole run: fall back to the
            # persistence guess (the checkpoint close) if it can be read,
            # otherwise skip the case.  A missing output is scored as
            # persistence anyway, but a crash risks the other cases too.
            try:
                with open(path, encoding="utf-8") as f:
                    case = json.load(f)
                price = case["markets"][0]["prices"][-1]
            except Exception:
                continue
        out = {"forecasts": [{"case": case_id, "price": round(price, 2)}]}
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
