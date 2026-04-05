import json
import os
import threading
import time
import types
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
load_dotenv(Path(__file__).resolve().parent.parent / "1" / ".env", override=False)

# =========================
# === USER CONFIG
# =========================
API_KEY = os.environ.get("BYBIT_API_KEY", "")
API_SECRET = os.environ.get("BYBIT_API_SECRET", "")
CONFIG_PATH = Path("config.json")
BACKTEST_INTERVAL_SECONDS = 8 * 60 * 60

ML_LAYERS_CODE = 'from __future__ import annotations\n\nimport json\nimport math\nfrom dataclasses import dataclass\nfrom typing import Dict, Iterable, List, Optional, Sequence, Tuple\n\nimport numpy as np\nimport pandas as pd\n\n\n@dataclass(frozen=True)\nclass ParamSpec:\n    name: str\n    low: float\n    high: float\n    kind: str  # "int" or "float"\n\n\nclass ParameterSpace:\n    def __init__(self, specs: Sequence[ParamSpec]) -> None:\n        self.specs = list(specs)\n\n    def sample_random(self, rng: np.random.Generator) -> Dict[str, float]:\n        params: Dict[str, float] = {}\n        for spec in self.specs:\n            if spec.kind == "int":\n                params[spec.name] = int(rng.integers(int(spec.low), int(spec.high) + 1))\n            else:\n                params[spec.name] = float(rng.uniform(spec.low, spec.high))\n        return params\n\n    def clamp(self, params: Dict[str, float]) -> Dict[str, float]:\n        clamped: Dict[str, float] = {}\n        for spec in self.specs:\n            val = float(params[spec.name])\n            val = min(max(val, spec.low), spec.high)\n            if spec.kind == "int":\n                val = int(round(val))\n                val = max(int(spec.low), min(int(spec.high), val))\n            clamped[spec.name] = val\n        return clamped\n\n\ndef _normal_pdf(x: float, mean: float, std: float) -> float:\n    std = max(std, 1e-6)\n    z = (x - mean) / std\n    return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))\n\n\nclass TPEModel:\n    """\n    Lightweight Tree-Parzen Estimator:\n    - Split history into good/bad buckets by score quantile\n    - Fit per-parameter Gaussians for each bucket\n    - Sample from good model, then pick candidates with best likelihood ratio\n    """\n\n    def __init__(self, space: ParameterSpace, gamma: float = 0.2) -> None:\n        self.space = space\n        self.gamma = gamma\n\n    def suggest(self, history: List[dict], rng: np.random.Generator, n_candidates: int = 32) -> Dict[str, float]:\n        if len(history) < 20:\n            return self.space.sample_random(rng)\n\n        sorted_hist = sorted(history, key=lambda x: x["score"], reverse=True)\n        cut = max(1, int(len(sorted_hist) * self.gamma))\n        good = sorted_hist[:cut]\n        bad = sorted_hist[cut:]\n        if not bad:\n            return self.space.sample_random(rng)\n\n        def bucket_stats(records: List[dict]) -> Dict[str, Tuple[float, float]]:\n            stats: Dict[str, Tuple[float, float]] = {}\n            for spec in self.space.specs:\n                vals = []\n                for r in records:\n                    params = r.get("params", {})\n                    if spec.name in params:\n                        vals.append(float(params[spec.name]))\n                if not vals:\n                    vals = [float(self.space.sample_random(rng)[spec.name]) for _ in range(8)]\n                mean = float(np.mean(vals))\n                std = float(np.std(vals) + 1e-6)\n                stats[spec.name] = (mean, std)\n            return stats\n\n        good_stats = bucket_stats(good)\n        bad_stats = bucket_stats(bad)\n\n        best_candidate = None\n        best_ratio = -math.inf\n        for _ in range(n_candidates):\n            candidate = {}\n            ratio = 1.0\n            for spec in self.space.specs:\n                mean_g, std_g = good_stats[spec.name]\n                mean_b, std_b = bad_stats[spec.name]\n                sampled = rng.normal(mean_g, std_g)\n                candidate[spec.name] = sampled\n                p_good = _normal_pdf(sampled, mean_g, std_g)\n                p_bad = _normal_pdf(sampled, mean_b, std_b)\n                ratio *= p_good / max(p_bad, 1e-12)\n            candidate = self.space.clamp(candidate)\n            if ratio > best_ratio:\n                best_ratio = ratio\n                best_candidate = candidate\n\n        if best_candidate is None:\n            return self.space.sample_random(rng)\n        return best_candidate\n\n\nclass SurrogateOptimizer:\n    def __init__(\n        self,\n        space: ParameterSpace,\n        history_path: str,\n        seed: int = 7,\n        explore_ratio: float = 0.15,\n    ) -> None:\n        self.space = space\n        self.history_path = history_path\n        self.rng = np.random.default_rng(seed)\n        self.explore_ratio = explore_ratio\n        self.tpe = TPEModel(space)\n        self.history: List[dict] = []\n        self._seen = set()\n        self._load_history()\n\n    def _load_history(self) -> None:\n        try:\n            with open(self.history_path, "r", encoding="utf-8") as f:\n                for line in f:\n                    line = line.strip()\n                    if not line:\n                        continue\n                    rec = json.loads(line)\n                    self.history.append(rec)\n                    self._seen.add(self._hash_params(rec["params"]))\n        except FileNotFoundError:\n            return\n\n    def _hash_params(self, params: Dict[str, float]) -> str:\n        parts = [f"{k}={params[k]}" for k in sorted(params)]\n        return "|".join(parts)\n\n    def suggest(self) -> Dict[str, float]:\n        if self.rng.random() < self.explore_ratio or len(self.history) < 20:\n            candidate = self.space.sample_random(self.rng)\n        else:\n            candidate = self.tpe.suggest(self.history, self.rng)\n        candidate = self.space.clamp(candidate)\n        attempts = 0\n        while self._hash_params(candidate) in self._seen and attempts < 25:\n            candidate = self.space.sample_random(self.rng)\n            candidate = self.space.clamp(candidate)\n            attempts += 1\n        return candidate\n\n    def record(self, params: Dict[str, float], metrics: Dict[str, float], score: float) -> None:\n        rec = {"params": params, "metrics": metrics, "score": float(score)}\n        self.history.append(rec)\n        self._seen.add(self._hash_params(params))\n        with open(self.history_path, "a", encoding="utf-8") as f:\n            f.write(json.dumps(rec))\n            f.write("\\n")\n\n    def top_results(self, n: int = 3) -> List[dict]:\n        return sorted(self.history, key=lambda x: x["score"], reverse=True)[:n]\n\n\ndef compute_regime_features(df, window: int = 100) -> Dict[str, float]:\n    df = df.tail(window)\n    returns = df["close"].pct_change().dropna()\n    if len(returns) < 5:\n        return {\n            "volatility": 0.0,\n            "trend_strength": 0.0,\n            "volume_z": 0.0,\n            "return_skew": 0.0,\n            "return_kurt": 0.0,\n        }\n    volatility = float(returns.std())\n    x = np.arange(len(df))\n    y = df["close"].to_numpy()\n    slope = float(np.polyfit(x, y, 1)[0]) if len(df) > 2 else 0.0\n    trend_strength = slope / max(df["close"].mean(), 1e-6)\n    volume = df["volume"].to_numpy()\n    volume_z = float((volume[-1] - volume.mean()) / max(volume.std(), 1e-6))\n    return_skew = float(returns.skew())\n    return_kurt = float(returns.kurtosis())\n    return {\n        "volatility": volatility,\n        "trend_strength": trend_strength,\n        "volume_z": volume_z,\n        "return_skew": return_skew,\n        "return_kurt": return_kurt,\n    }\n\n\nclass RegimeDetector:\n    def __init__(self, feature_order: Sequence[str], centers: np.ndarray, trade_cluster: int) -> None:\n        self.feature_order = list(feature_order)\n        self.centers = centers\n        self.trade_cluster = trade_cluster\n\n    @staticmethod\n    def fit(features: List[Dict[str, float]], k: int = 2, seed: int = 7) -> "RegimeDetector":\n        rng = np.random.default_rng(seed)\n        if not features:\n            feature_order = ["volatility", "trend_strength", "volume_z", "return_skew", "return_kurt"]\n            centers = np.zeros((k, len(feature_order)))\n            return RegimeDetector(feature_order, centers, trade_cluster=0)\n        feature_order = list(features[0].keys())\n        data = np.array([[f[name] for name in feature_order] for f in features], dtype=float)\n        centers = data[rng.choice(len(data), size=min(k, len(data)), replace=False)]\n        for _ in range(20):\n            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)\n            labels = np.argmin(distances, axis=1)\n            new_centers = []\n            for idx in range(centers.shape[0]):\n                cluster = data[labels == idx]\n                if len(cluster) == 0:\n                    new_centers.append(centers[idx])\n                else:\n                    new_centers.append(cluster.mean(axis=0))\n            new_centers = np.vstack(new_centers)\n            if np.allclose(new_centers, centers):\n                break\n            centers = new_centers\n        # Trade cluster: most negative trend + above-median volatility\n        trend_idx = feature_order.index("trend_strength")\n        vol_idx = feature_order.index("volatility")\n        trend_vals = centers[:, trend_idx]\n        vol_vals = centers[:, vol_idx]\n        trade_cluster = int(np.argmin(trend_vals))\n        if vol_vals[trade_cluster] < np.median(vol_vals):\n            trade_cluster = int(np.argmax(vol_vals))\n        return RegimeDetector(feature_order, centers, trade_cluster=trade_cluster)\n\n    def classify(self, features: Dict[str, float]) -> str:\n        vec = np.array([features[name] for name in self.feature_order], dtype=float)\n        distances = np.linalg.norm(self.centers - vec[None, :], axis=1)\n        cluster = int(np.argmin(distances))\n        return "TRADE" if cluster == self.trade_cluster else "NO_TRADE"\n\n\nclass AnomalyDetector:\n    def __init__(self, feature_order: Sequence[str], mean: np.ndarray, std: np.ndarray, z_thresh: float) -> None:\n        self.feature_order = list(feature_order)\n        self.mean = mean\n        self.std = np.maximum(std, 1e-6)\n        self.z_thresh = z_thresh\n\n    @staticmethod\n    def fit(features: List[Dict[str, float]], z_thresh: float = 3.5) -> "AnomalyDetector":\n        if not features:\n            feature_order = ["volatility", "trend_strength", "volume_z", "return_skew", "return_kurt"]\n            return AnomalyDetector(feature_order, np.zeros(len(feature_order)), np.ones(len(feature_order)), z_thresh)\n        feature_order = list(features[0].keys())\n        data = np.array([[f[name] for name in feature_order] for f in features], dtype=float)\n        mean = data.mean(axis=0)\n        std = data.std(axis=0)\n        return AnomalyDetector(feature_order, mean, std, z_thresh)\n\n    def is_anomaly(self, features: Dict[str, float]) -> bool:\n        vec = np.array([features[name] for name in self.feature_order], dtype=float)\n        z = np.abs((vec - self.mean) / self.std)\n        return bool(np.any(z > self.z_thresh))\n\n'
BACKTESTER_CODE = '''import os
import multiprocessing as _mp
import time
import math
import json
import importlib.util
import hashlib
import hmac
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from ta.momentum import RSIIndicator

from ml_layers import (
    AnomalyDetector,
    ParameterSpace,
    ParamSpec,
    RegimeDetector,
    SurrogateOptimizer,
    compute_regime_features,
)
def compute_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def compute_ma(series: pd.Series, period: int, ma_type: int) -> pd.Series:
    if ma_type == 1:
        return series.ewm(span=period, adjust=False).mean()
    if ma_type == 2:
        return compute_wma(series, period)
    return series.rolling(period).mean()


def get_price_source(df: pd.DataFrame, source_type: int) -> pd.Series:
    if source_type == 1:
        return (df["high"] + df["low"]) / 2.0
    if source_type == 2:
        return (df["high"] + df["low"] + df["close"]) / 3.0
    return df["close"]

# =========================
# === CONFIG
# =========================
SYMBOL = "ESPUSDT"
CATEGORY = "linear"      # Bybit USDT perpetuals
INTERVAL = "15"         # default interval (updated after multi-interval backtest)
INTERVALS = ["5"]
DAYS = 14                # last 14 days
N_COMBOS_PER_INTERVAL = 1_000          # random combinations per interval
OPTIMIZER_HISTORY_PATH = "optimizer_history.jsonl"
OPTIMIZER_SEED = 7
OPTIMIZER_FOLDS = 3
MIN_OPTIMIZER_DAYS = 7
MAX_OPTIMIZER_DAYS = 7
REGIME_WINDOW = 120
ANOMALY_Z_THRESH = 3.5
BACKTEST_DATA_DIR = Path("backtest_data")

START_BALANCE = 30.0
POSITION_SIZE = 0.60

TAKER_FEE = 0.00075      # 0.075%
SLIPPAGE = 0.0001    # 0.01%
TARGET_ROI_PCT = 0.04      # 4% ROI target
TAKE_PROFIT_PCT = 0.0033     # Take-profit % (0.33% — only exit mechanism)
TARGET_ROI_MIN_PCT = 0.005
TARGET_ROI_MAX_PCT = 0.10

API_URL = "https://api.bybit.com/v5/market/kline"
API_LIMIT = 200
REQUEST_TIMEOUT = 12
MAX_RETRIES = 6
RATE_LIMIT_CODE = 10006
RATE_LIMIT_SLEEP_BASE = 2

# =========================
# === API AUTH
# =========================



API_KEY = os.environ.get("BYBIT_API_KEY", "")
API_SECRET = os.environ.get("BYBIT_API_SECRET", "")

RECV_WINDOW = "5000"

PRIVATE_BASE_URL = "https://api.bybit.com"

BACKTEST_INTERVAL_SECONDS = 8 * 60 * 60


# =========================
# === DATA FETCH (ROBUST)
# =========================
def _rate_limit_sleep(attempt: int) -> None:
    delay = min(RATE_LIMIT_SLEEP_BASE * attempt * attempt, 30)
    time.sleep(delay)


def _is_rate_limit_response(payload: dict) -> bool:
    return payload.get("retCode") == RATE_LIMIT_CODE


def _safe_get(url, params):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            j = r.json()
            if isinstance(j, dict) and _is_rate_limit_response(j):
                last_err = RuntimeError(
                    f"Rate limited retCode={j.get('retCode')} retMsg={j.get('retMsg')}"
                )
                _rate_limit_sleep(attempt)
                continue
            return j
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * attempt, 6))
    raise RuntimeError(f"HTTP/JSON failed after {MAX_RETRIES} retries: {last_err}")


def _sign_request(timestamp_ms: str, api_key: str, recv_window: str, query_string: str) -> str:
    payload = f"{timestamp_ms}{api_key}{recv_window}{query_string}"
    return hmac.new(API_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _safe_private_get(path: str, params: dict) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            timestamp_ms = str(int(time.time() * 1000))
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = _sign_request(timestamp_ms, API_KEY, RECV_WINDOW, query_string)
            headers = {
                "X-BAPI-API-KEY": API_KEY,
                "X-BAPI-SIGN": signature,
                "X-BAPI-TIMESTAMP": timestamp_ms,
                "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            }
            url = f"{PRIVATE_BASE_URL}{path}"
            r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            j = r.json()
            if isinstance(j, dict) and _is_rate_limit_response(j):
                last_err = RuntimeError(
                    f"Rate limited retCode={j.get('retCode')} retMsg={j.get('retMsg')}"
                )
                _rate_limit_sleep(attempt)
                continue
            return j
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * attempt, 6))
    raise RuntimeError(f"Private HTTP/JSON failed after {MAX_RETRIES} retries: {last_err}")


def fetch_bybit_ohlcv(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    Paginate backwards using 'end' in ms until we have ~days of candles or Bybit runs out.
    Returns dataframe indexed by UTC timestamp with columns open/high/low/close/volume.
    """
    end_ms = int(time.time() * 1000)
    start_limit_ms = end_ms - int(days * 24 * 60 * 60 * 1000)

    rows_all = []
    seen_ends = set()

    print(f"Downloading {days}d of {interval}m klines for {symbol} from Bybit...")

    while True:
        if end_ms in seen_ends:
            # Prevent infinite loop if API keeps returning same page
            break
        seen_ends.add(end_ms)

        params = {
            "category": CATEGORY,
            "symbol": symbol.upper(),
            "interval": interval,
            "end": end_ms,
            "limit": API_LIMIT,
        }

        j = _safe_get(API_URL, params)

        if j.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")

        page = (j.get("result") or {}).get("list") or []
        if not page:
            break

        rows_all.extend(page)

        oldest_ts_ms = int(page[-1][0])
        if oldest_ts_ms <= start_limit_ms:
            break

        end_ms = oldest_ts_ms - 1
        time.sleep(0.12)  # gentle pacing

    if not rows_all:
        raise RuntimeError(f"No candles returned for {symbol}. API returned empty dataset.")

    df = pd.DataFrame(rows_all)

    # Bybit rows usually: [startTime, open, high, low, close, volume, turnover]
    # Keep only first 6 safely.
    if df.shape[1] < 6:
        raise RuntimeError(f"Unexpected kline schema: got {df.shape[1]} columns, expected >= 6")

    df = df.iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # Deduplicate + sort ascending
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    df.set_index("timestamp", inplace=True)

    # Trim strictly to last N days (in case extra fetched)
    cutoff = df.index.max() - pd.Timedelta(days=days)
    df = df[df.index >= cutoff].copy()

    run_tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    BACKTEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    interval_label = str(interval).replace(".", "_")
    safe_symbol = symbol.replace("/", "_")
    csv_path = BACKTEST_DATA_DIR / f"{safe_symbol}_{interval_label}m_{days}d_{run_tag}.csv"
    df.reset_index().to_csv(csv_path, index=False)
    print(f"Downloaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
    print(f"Saved backtest data to {csv_path}")

    reloaded = pd.read_csv(csv_path)
    reloaded["timestamp"] = pd.to_datetime(reloaded["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        reloaded[c] = reloaded[c].astype(float)
    reloaded.set_index("timestamp", inplace=True)
    return reloaded


def _optimizer_window_days(symbol: str, interval: str, days: int) -> int:
    if symbol.upper().startswith("BTC") and str(interval) == "5":
        return max(days, MAX_OPTIMIZER_DAYS)
    return max(days, MIN_OPTIMIZER_DAYS)


def fetch_fee_rates(symbol: str) -> dict:
    params = {"category": CATEGORY, "symbol": symbol.upper()}
    j = _safe_private_get("/v5/account/fee-rate", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Fee-rate error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    rows = (j.get("result") or {}).get("list") or []
    if not rows:
        raise RuntimeError("Fee-rate list empty; cannot determine fees.")
    return rows[0]


def fetch_position_leverage(symbol: str) -> float:
    params = {"category": CATEGORY, "symbol": symbol.upper()}
    j = _safe_private_get("/v5/position/list", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Position list error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    positions = (j.get("result") or {}).get("list") or []
    if not positions:
        return 1.0
    leverage = float(positions[0].get("leverage", 1.0))
    return max(leverage, 1.0)


def fetch_wallet_balance(coin: str = "USDT") -> float:
    params = {"accountType": "UNIFIED", "coin": coin}
    j = _safe_private_get("/v5/account/wallet-balance", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Wallet balance error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    rows = (j.get("result") or {}).get("list") or []
    if not rows:
        raise RuntimeError("Wallet balance list empty; cannot determine balance.")
    coins = rows[0].get("coin") or []
    for entry in coins:
        if entry.get("coin") == coin:
            return float(entry.get("walletBalance", 0.0))
    raise RuntimeError(f"Coin {coin} not found in wallet balance response.")


# =========================
# === NON-REPAINT BACKTEST
# =========================
def backtest_no_lookahead(
    df: pd.DataFrame,
    params: tuple,
    initial_equity: float = START_BALANCE,
    taker_fee: float = TAKER_FEE,
    maker_fee: float = 0.0,
    slippage: float = SLIPPAGE,
    position_size: float = POSITION_SIZE,
    leverage: float = 1.0,
):
    """
    No repaint / no lookahead:
      - Indicators are shifted by 1 (computed using completed bar i-1 and earlier)
      - Entry executes as post-only limit at OPEN of bar i if signal was true on bar i-1
      - Exits are evaluated using bar i OHLC only after entry is already open
    Short-only strategy with separate short entry + exit params.
    """

    (
        # SHORT ENTRY PARAMS
        s_fast, s_mid, s_slow, s_rsi_len, ema_len, ma_type, rsi_source, ema_source,
        volume_filter, volume_ma_period, volume_threshold,
        avoid_first_hours, avoid_last_hours, s_rsi_trig,
        # ROI PARAMS
        target_roi_pct,
    ) = params

    d = df.copy()
    take_profit_pct = TAKE_PROFIT_PCT  # 0.33% take-profit — only exit mechanism

    # Compute indicators
    ma_type = int(ma_type)
    rsi_source = int(rsi_source)
    ema_source = int(ema_source)
    volume_ma_period = int(volume_ma_period)
    avoid_first_hours = int(avoid_first_hours)
    avoid_last_hours = int(avoid_last_hours)

    price_source = get_price_source(d, rsi_source)
    ema_source_series = get_price_source(d, ema_source)

    d["s_fast"] = compute_ma(d["close"], int(s_fast), ma_type)
    d["s_mid"]  = compute_ma(d["close"], int(s_mid),  ma_type)
    d["s_slow"] = compute_ma(d["close"], int(s_slow), ma_type)
    d["s_rsi"]  = RSIIndicator(price_source, int(s_rsi_len)).rsi()
    d["ema"]    = ema_source_series.ewm(span=int(ema_len), adjust=False).mean()

    _vmp = max(int(volume_ma_period), 1)
    d["volume_ma"]  = d["volume"].rolling(_vmp).mean()
    d["hour"] = d.index.hour

    # Shift everything 1 bar to avoid using current bar info
    for col in ["s_fast", "s_mid", "s_slow", "s_rsi", "ema", "volume_ma", "hour"]:
        d[col] = d[col].shift(1)

    # Signals evaluated on bar i-1, traded on bar i open
    d["ema_trend_up"] = d["ema"] > d["ema"].shift(1)
    volume_ok = True
    if int(volume_filter) == 1:
        volume_ok = d["volume"].shift(1) > (d["volume_ma"] * float(volume_threshold))
    time_ok = True
    if avoid_first_hours > 0 or avoid_last_hours > 0:
        hour_prev = d["hour"]
        time_ok = (hour_prev >= avoid_first_hours) & (hour_prev < (24 - avoid_last_hours))

    d["short_signal"] = (
        (d["s_fast"] < d["s_mid"])
        & (d["s_mid"] < d["s_slow"])
        & (d["s_rsi"] > s_rsi_trig)
        & (~d["ema_trend_up"])
        & volume_ok
        & time_ok
    )

    equity = float(initial_equity)
    position = 0  # 0 flat, -1 short
    entry_price = 0.0

    equity_curve = []
    trade_pnls_per_unit = []   # per-unit PnL in price terms (we scale by size)
    trade_pnls_usdt = []
    gross_profit = 0.0
    gross_loss = 0.0
    trade_wins = 0
    trades = 0
    max_trade_loss = 0.0
    entry_index = None
    position_units = 0.0
    position_size_units = 0.0

    # Pre-extract all columns to numpy arrays — eliminates pandas .iloc[i] overhead in loop
    _arr_open   = d["open"].to_numpy(dtype="float64")
    _arr_high   = d["high"].to_numpy(dtype="float64")
    _arr_low    = d["low"].to_numpy(dtype="float64")
    _arr_sig    = d["short_signal"].to_numpy(dtype="bool")

    # Start after enough warmup bars
    start_i = 2 + max(int(s_slow), int(s_rsi_len), int(ema_len), int(volume_ma_period))

    for i in range(start_i, len(d)):
        o = _arr_open[i]
        h = _arr_high[i]
        l = _arr_low[i]

        # -------- EXIT (on bar i OHLC, only if we already have a position) --------
        if position != 0:
            exited = False
            pnl_per_unit = 0.0
            if position == -1:
                take_profit = entry_price * (1.0 - take_profit_pct)
                if l <= take_profit:
                    exit_px = take_profit * (1.0 + slippage)
                    exit_px = exit_px * (1.0 - taker_fee)
                    pnl_per_unit = entry_price - exit_px
                    exited = True

            if exited:
                size_units = position_size_units * position_units
                equity += pnl_per_unit * size_units
                trade_pnl_usdt = pnl_per_unit * size_units
                trade_pnls_per_unit.append(pnl_per_unit)
                trade_pnls_usdt.append(trade_pnl_usdt)
                trades += 1
                if pnl_per_unit > 0:
                    trade_wins += 1
                    gross_profit += trade_pnl_usdt
                    result_label = "WIN"
                else:
                    max_trade_loss = min(max_trade_loss, trade_pnl_usdt)
                    gross_loss += trade_pnl_usdt
                    result_label = "LOSS"
                if entry_index is not None:
                    print(f"[trade] {result_label} entry_bar={entry_index} exit_bar={i} pnl={trade_pnl_usdt:.4f}")
                position = 0
                entry_price = 0.0
                entry_index = None
                position_units = 0.0
                position_size_units = 0.0

        # -------- ENTRY (signal on i-1 => enter at open i, post-only maker) --------
        if position == 0:
            if bool(_arr_sig[i - 1]):
                # Short entry: post-only limit modeled at open with maker fee
                entry_px = o * (1.0 - maker_fee)
                position = -1
                entry_price = entry_px
                position_units = 1.0
                position_size_units = (equity * position_size * leverage) / entry_price
                entry_index = i - 1
                print(f"[entry] bar={entry_index} price={entry_px:.4f}")

        equity_curve.append(equity)

        # Hard stop if equity goes to ~0
        if equity <= 0.5:
            break

    if trades == 0:
        return None

    pnl_arr = np.array(trade_pnls_per_unit)
    win_rate = trade_wins / trades

    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    gross_loss_abs = abs(gross_loss)
    if gross_loss_abs > 0:
        profit_factor = float(gross_profit / gross_loss_abs)
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    expectancy = float(np.mean(trade_pnls_usdt)) if trade_pnls_usdt else 0.0

    return {
        "final_equity": float(equity),
        "net_profit": float(equity - initial_equity),
        "return_pct": float((equity / initial_equity - 1.0) * 100.0),
        "trades": int(trades),
        "wins": int(trade_wins),
        "win_rate": float(win_rate),
        "avg_pnl_per_unit": float(pnl_arr.mean()),
        "max_dd": max_dd,
        "max_loss": float(max_trade_loss),
        "max_loss_pct": float(max_trade_loss / initial_equity),
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "target_roi_pct": float(target_roi_pct),
        "equity_curve": equity_curve,
    }


# =========================
# === PARAM SAMPLING
# =========================
PARAM_SPACE = ParameterSpace(
    [
        ParamSpec("s_fast", 5, 20, "int"),
        ParamSpec("s_mid", 12, 60, "int"),
        ParamSpec("s_slow", 30, 140, "int"),
        ParamSpec("s_rsi_len", 7, 21, "int"),
        ParamSpec("ema_len", 20, 200, "int"),
        ParamSpec("ma_type", 0, 2, "int"),
        ParamSpec("rsi_source", 0, 2, "int"),
        ParamSpec("ema_source", 0, 2, "int"),
        ParamSpec("volume_filter", 0, 1, "int"),
        ParamSpec("volume_ma_period", 10, 50, "int"),
        ParamSpec("volume_threshold", 1.0, 2.5, "float"),
        ParamSpec("avoid_first_hours", 0, 4, "int"),
        ParamSpec("avoid_last_hours", 0, 4, "int"),
        ParamSpec("s_rsi_trig", 45, 65, "float"),
        ParamSpec("target_roi_pct", TARGET_ROI_MIN_PCT, TARGET_ROI_MAX_PCT, "float"),
    ]
)


def _normalize_params(params: dict) -> tuple:
    s_fast = int(params["s_fast"])
    s_mid = max(int(params["s_mid"]), s_fast + 2)
    s_slow = max(int(params["s_slow"]), s_mid + 5)
    s_rsi_len = int(params["s_rsi_len"])
    ema_len = int(params["ema_len"])
    ma_type = int(params["ma_type"])
    rsi_source = int(params["rsi_source"])
    ema_source = int(params["ema_source"])
    volume_filter = int(params["volume_filter"])
    volume_ma_period = int(params["volume_ma_period"])
    volume_threshold = float(params["volume_threshold"])
    avoid_first_hours = int(params["avoid_first_hours"])
    avoid_last_hours = int(params["avoid_last_hours"])
    s_rsi_trig = float(params["s_rsi_trig"])
    target_roi_pct = float(params.get("target_roi_pct", TARGET_ROI_PCT))
    return (
        s_fast,
        s_mid,
        s_slow,
        s_rsi_len,
        ema_len,
        ma_type,
        rsi_source,
        ema_source,
        volume_filter,
        volume_ma_period,
        volume_threshold,
        avoid_first_hours,
        avoid_last_hours,
        s_rsi_trig,
        target_roi_pct,
    )


def _fold_slices(df: pd.DataFrame, folds: int) -> List[pd.DataFrame]:
    n = len(df)
    if n < folds * 50:
        return []
    indices = np.array_split(np.arange(n), folds)
    return [df.iloc[idx] for idx in indices if len(idx) > 10]


def _aggregate_fold_metrics(results: List[dict]) -> dict:
    returns = [r["return_pct"] for r in results]
    win_rates = [r["win_rate"] for r in results]
    trades = [r["trades"] for r in results]
    max_losses = [r["max_loss"] for r in results]
    max_loss_pcts = [r["max_loss_pct"] for r in results]
    max_dds = [r["max_dd"] for r in results]
    profit_factors = [r["profit_factor"] for r in results]
    expectancies = [r["expectancy"] for r in results]
    net_profits = [r["net_profit"] for r in results]
    target_rois = [r["target_roi_pct"] for r in results]
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "worst_dd": float(min(max_dds)),
        "mean_win_rate": float(np.mean(win_rates)),
        "mean_trades": float(np.mean(trades)),
        "total_trades": int(sum(trades)),
        "worst_max_loss": float(min(max_losses)),
        "worst_max_loss_pct": float(min(max_loss_pcts)),
        "profit_factor": float(min(profit_factors)),
        "expectancy": float(np.mean(expectancies)),
        "net_profit": float(np.mean(net_profits)),
        "mean_target_roi_pct": float(np.mean(target_rois)),
    }


def draw_equity_curve_terminal(equity_curve, width: int = 60) -> None:
    if not equity_curve:
        print("Equity curve: (empty)")
        return

    if importlib.util.find_spec("plotext") is not None:
        import plotext as plt

        data = np.array(equity_curve, dtype=float)
        if len(data) > width:
            idx = np.linspace(0, len(data) - 1, width).astype(int)
            data = data[idx]

        plt.clear_figure()
        plt.plot(data)
        plt.title("Equity Curve (terminal)")
        plt.ylabel("Equity")
        plt.xlabel("Bars")
        plt.show()
        return

    data = np.array(equity_curve, dtype=float)
    if len(data) > width:
        idx = np.linspace(0, len(data) - 1, width).astype(int)
        data = data[idx]
    else:
        width = len(data)

    min_val = float(np.min(data))
    max_val = float(np.max(data))
    span = max_val - min_val
    if span == 0:
        span = 1.0

    blocks = "▁▂▃▄▅▆▇█"
    levels = len(blocks) - 1
    spark = []
    for val in data:
        level = int(round((val - min_val) / span * levels))
        level = max(0, min(levels, level))
        spark.append(blocks[level])

    print("Equity Curve (sparkline)")
    print("".join(spark))
    print(f"min={min_val:.4f} max={max_val:.4f}")


# =========================
# === MAIN
# =========================
def _interval_history_path(interval: str) -> str:
    interval_label = str(interval).replace(".", "_")
    return f"optimizer_history_{interval_label}m.jsonl"


_N_WORKERS = max(1, _mp.cpu_count() - 1)
_BATCH_SIZE = _N_WORKERS * 4

# Shared eval state populated before pool creation; inherited by forked workers.
_eval_state: dict = {}


def _eval_combo(params):
    st = _eval_state
    results = []
    for fold_df in st["fold_slices"]:
        res = backtest_no_lookahead(
            fold_df,
            params,
            initial_equity=st["initial_equity"],
            taker_fee=st["taker_fee"],
            maker_fee=st["maker_fee"],
            slippage=st["slippage"],
            position_size=st["position_size"],
            leverage=st["leverage"],
        )
        if res is not None:
            results.append(res)
    return results if results else None


def run_backtest_cycle():
    fee_rates = fetch_fee_rates(SYMBOL)
    taker_fee = float(fee_rates.get("takerFeeRate", TAKER_FEE))
    maker_fee = float(fee_rates.get("makerFeeRate", 0.0))
    leverage = fetch_position_leverage(SYMBOL)
    actual_balance = fetch_wallet_balance("USDT")

    best_overall_score = -math.inf
    best_overall_params = None
    best_overall = None
    best_overall_interval = None
    best_overall_days = None
    best_overall_df = None

    for interval in INTERVALS:
        optimizer_days = _optimizer_window_days(SYMBOL, interval, DAYS)
        df = fetch_bybit_ohlcv(SYMBOL, interval, optimizer_days)

        # Basic sanity
        if len(df) < 200:
            print(f"[warn] too few candles fetched ({len(df)}) for {interval}m; skipping.")
            continue

        optimizer = SurrogateOptimizer(
            PARAM_SPACE,
            _interval_history_path(interval),
            seed=OPTIMIZER_SEED,
        )

        best_score = -math.inf
        best_params = None
        fold_slices = _fold_slices(df, OPTIMIZER_FOLDS)
        if not fold_slices:
            fold_slices = [df]

        print("")
        print(
            "Running {:,} surrogate-guided backtests for {}m (no lookahead, start={:.2f} USDT)...".format(
                N_COMBOS_PER_INTERVAL,
                interval,
                actual_balance,
            )
        )
        print("")

        _eval_state["fold_slices"] = fold_slices
        _eval_state["initial_equity"] = actual_balance
        _eval_state["taker_fee"] = taker_fee
        _eval_state["maker_fee"] = maker_fee
        _eval_state["slippage"] = SLIPPAGE
        _eval_state["position_size"] = POSITION_SIZE
        _eval_state["leverage"] = leverage

        remaining = N_COMBOS_PER_INTERVAL
        done = 0
        with _mp.get_context("fork").Pool(_N_WORKERS) as pool:
            while remaining > 0:
                batch_size = min(_BATCH_SIZE, remaining)
                batch_dicts = [optimizer.suggest() for _ in range(batch_size)]
                batch_params = [_normalize_params(d) for d in batch_dicts]
                batch_results = pool.map_async(_eval_combo, batch_params).get(timeout=120)
                for params_dict, params, fold_results in zip(batch_dicts, batch_params, batch_results):
                    if fold_results is None:
                        continue
                    metrics = _aggregate_fold_metrics(fold_results)
                    sc = float(metrics.get("mean_return", 0.0))
                    optimizer.record(params_dict, metrics, sc)
                    if sc > best_score:
                        best_score = sc
                        best_params = params
                remaining -= batch_size
                done += batch_size
                print(f"[opt] {done}/{N_COMBOS_PER_INTERVAL} combos tested", flush=True)

        if best_params is None:
            print(f"[warn] no configurations evaluated for {interval}m.")
            continue

        best = backtest_no_lookahead(
            df,
            best_params,
            initial_equity=actual_balance,
            taker_fee=taker_fee,
            maker_fee=maker_fee,
            slippage=SLIPPAGE,
            position_size=POSITION_SIZE,
            leverage=leverage,
        )
        if best is None:
            print(f"[warn] best configuration failed on full window for {interval}m.")
            continue

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_params = best_params
            best_overall = best
            best_overall_interval = interval
            best_overall_days = optimizer_days
            best_overall_df = df

    if best_overall_params is None or best_overall is None:
        print("No configurations evaluated.")
        return

    global INTERVAL
    best_score = best_overall_score
    best_params = best_overall_params
    best = best_overall
    optimizer_days = best_overall_days
    df = best_overall_df
    INTERVAL = str(best_overall_interval)

    # ===== Terminal summary =====
    print("\\n================= BEST CONFIG =================")
    print(f"Symbol            : {SYMBOL}")
    print(f"Timeframe         : {INTERVAL}m")
    print(f"Window            : last {optimizer_days} days")
    print(f"Start balance     : {actual_balance:.2f} USDT")
    print(f"Final balance     : {best['final_equity']:.4f} USDT")
    print(f"Net profit        : {best['net_profit']:.4f} USDT")
    print(f"Return            : {best['return_pct']:.2f}%")
    print(f"Trades            : {best['trades']} (wins {best['wins']})")
    print(f"Win rate          : {best['win_rate']*100:.2f}%")
    print(f"Max drawdown      : {best['max_dd']*100:.2f}%")
    print(f"Max loss (trade)  : {best['max_loss']:.4f} USDT")
    print(f"Score (edge)      : {best_score:.6f}")
    (
        s_fast,
        s_mid,
        s_slow,
        s_rsi_len,
        ema_len,
        ma_type,
        rsi_source,
        ema_source,
        volume_filter,
        volume_ma_period,
        volume_threshold,
        avoid_first_hours,
        avoid_last_hours,
        s_rsi_trig,
        target_roi_pct,
    ) = best_params
    print(
        "Fees/slippage     : taker_fee=%.5f  maker_fee=%.5f  slippage=%.5f  pos=%.2f  leverage=%.2f"
        % (taker_fee, maker_fee, SLIPPAGE, POSITION_SIZE, leverage)
    )
    print(f"Target ROI pct   : {target_roi_pct:.4f}")
    print(f"Take profit pct  : {TAKE_PROFIT_PCT:.4f}")
    print("\\n--- Params ---")

    ma_type_label = {0: "SMA", 1: "EMA", 2: "WMA"}.get(ma_type, "SMA")
    print(
        "SHORT entry: fast=%d mid=%d slow=%d rsi_len=%d ema_len=%d ma_type=%s "
        "rsi_source=%d ema_source=%d rsi_trig=%.2f"
        % (s_fast, s_mid, s_slow, s_rsi_len, ema_len, ma_type_label, rsi_source, ema_source, s_rsi_trig)
    )
    print("================================================\\n")

    feature_history = []
    for idx in range(REGIME_WINDOW, len(df) + 1, REGIME_WINDOW):
        window_df = df.iloc[idx - REGIME_WINDOW:idx]
        feature_history.append(compute_regime_features(window_df, REGIME_WINDOW))
    regime_detector = RegimeDetector.fit(feature_history)
    anomaly_detector = AnomalyDetector.fit(feature_history, z_thresh=ANOMALY_Z_THRESH)

    top_results = optimizer.top_results(3)
    parameter_sets = []
    for idx, rec in enumerate(top_results, start=1):
        p = _normalize_params(rec["params"])
        (
            t_fast,
            t_mid,
            t_slow,
            t_rsi_len,
            t_ema_len,
            t_ma_type,
            t_rsi_source,
            t_ema_source,
            t_volume_filter,
            t_volume_ma_period,
            t_volume_threshold,
            t_avoid_first_hours,
            t_avoid_last_hours,
            t_rsi_trig,
            t_target_roi,
        ) = p
        parameter_sets.append(
            {
                "name": f"set_{idx}",
                "strategy": {
                    "short_entry": {
                        "fast": t_fast,
                        "mid": t_mid,
                        "slow": t_slow,
                        "rsi_len": t_rsi_len,
                        "ema_len": t_ema_len,
                        "ma_type": {0: "SMA", 1: "EMA", 2: "WMA"}.get(t_ma_type, "SMA"),
                        "rsi_source": t_rsi_source,
                        "ema_source": t_ema_source,
                        "rsi_trig": t_rsi_trig,
                        "volume_filter": t_volume_filter,
                        "volume_ma_period": t_volume_ma_period,
                        "volume_threshold": t_volume_threshold,
                        "avoid_first_hours": t_avoid_first_hours,
                        "avoid_last_hours": t_avoid_last_hours,
                    },
                    "short_exit": {},
                },
                "target_roi_pct": t_target_roi,
            }
        )

    if feature_history:
        vol_threshold = float(np.median([f["volatility"] for f in feature_history]))
        trend_threshold = float(np.median([f["trend_strength"] for f in feature_history]))
    else:
        vol_threshold = 0.0
        trend_threshold = 0.0

    selector_tree = {
        "feature": "volatility",
        "threshold": vol_threshold,
        "left": {"set": "set_1"},
        "right": {
            "feature": "trend_strength",
            "threshold": trend_threshold,
            "left": {"set": "set_1"},
            "right": {"set": "set_2" if len(parameter_sets) > 1 else "set_1"},
        },
    }

    config = {
        "symbol": SYMBOL,
        "category": CATEGORY,
        "interval": INTERVAL,
        "days": optimizer_days,
        "balance": actual_balance,
        "position_size": POSITION_SIZE,
        "leverage": leverage,
        "fees": {
            "taker_fee": taker_fee,
            "maker_fee": maker_fee,
        },
        "slippage": SLIPPAGE,
        "target_roi_pct": target_roi_pct,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "strategy": {
            "short_entry": {
                "fast": s_fast,
                "mid": s_mid,
                "slow": s_slow,
                "rsi_len": s_rsi_len,
                "ema_len": ema_len,
                "ma_type": ma_type_label,
                "rsi_source": rsi_source,
                "ema_source": ema_source,
                "rsi_trig": s_rsi_trig,
                "volume_filter": volume_filter,
                "volume_ma_period": volume_ma_period,
                "volume_threshold": volume_threshold,
                "avoid_first_hours": avoid_first_hours,
                "avoid_last_hours": avoid_last_hours,
            },
            "short_exit": {},
        },
        "parameter_sets": parameter_sets,
        "regime_model": {
            "feature_order": regime_detector.feature_order,
            "centers": regime_detector.centers.tolist(),
            "trade_cluster": regime_detector.trade_cluster,
            "window": REGIME_WINDOW,
        },
        "regime_selector": {
            "feature_order": regime_detector.feature_order,
            "tree": selector_tree,
        },
        "anomaly_model": {
            "feature_order": anomaly_detector.feature_order,
            "mean": anomaly_detector.mean.tolist(),
            "std": anomaly_detector.std.tolist(),
            "z_thresh": anomaly_detector.z_thresh,
            "window": REGIME_WINDOW,
        },
    }

    for path in ("candidate_pool.json", "config.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\\n")

    draw_equity_curve_terminal(best["equity_curve"])

    return best


if __name__ == "__main__":
    while True:
        run_backtest_cycle()
        time.sleep(BACKTEST_INTERVAL_SECONDS)
'''
LIVE_CODE = """import time
import json
import hashlib
import hmac
import os
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import requests
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

# =========================
# === API AUTH
# =========================

API_KEY = os.environ.get("BYBIT_API_KEY", "")
API_SECRET = os.environ.get("BYBIT_API_SECRET", "")
RECV_WINDOW = "5000"
PRIVATE_BASE_URL = "https://api.bybit.com"
PUBLIC_BASE_URL = "https://api.bybit.com"

REQUEST_TIMEOUT = 12
MAX_RETRIES = 6
API_LIMIT = 200
RATE_LIMIT_CODE = 10006
RATE_LIMIT_SLEEP_BASE = 2
TAKER_FEE_DEFAULT = 0.00075
MAKER_FEE_DEFAULT = 0.0
CONFIG_REFRESH_SECONDS = 4 * 60 * 60
TARGET_ROI_PCT_DEFAULT = 0.04
TAKE_PROFIT_PCT_DEFAULT = 0.0033  # Failsafe take-profit 0.33%
PROFIT_PAUSE_THRESHOLD = 1.50
PROFIT_PAUSE_CANDLES = 400
PROFIT_CHECK_INTERVAL = 60


def _restart_script():
    print("[profit] restarting script after pause")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def compute_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def compute_ma(series: pd.Series, period: int, ma_type: int) -> pd.Series:
    if ma_type == 1:
        return series.ewm(span=period, adjust=False).mean()
    if ma_type == 2:
        return compute_wma(series, period)
    return series.rolling(period).mean()


def get_price_source(df: pd.DataFrame, source_type: int) -> pd.Series:
    if source_type == 1:
        return (df["high"] + df["low"]) / 2.0
    if source_type == 2:
        return (df["high"] + df["low"] + df["close"]) / 3.0
    return df["close"]


@dataclass
class StrategyConfig:
    symbol: str
    category: str
    interval: str
    position_size: float
    leverage: float
    slippage: float
    short_fast: int
    short_mid: int
    short_slow: int
    short_rsi_len: int
    ema_len: int
    ma_type: int
    rsi_source: int
    ema_source: int
    short_rsi_trig: float
    take_profit_pct: float
    target_roi_pct: float
    volume_filter: int
    volume_ma_period: int
    volume_threshold: float
    avoid_first_hours: int
    avoid_last_hours: int


def _parse_ma_type(value: object) -> int:
    if isinstance(value, str):
        return {"SMA": 0, "EMA": 1, "WMA": 2}.get(value.upper(), 0)
    return int(value or 0)


def _sign_request(timestamp_ms: str, api_key: str, recv_window: str, query_string: str) -> str:
    payload = f"{timestamp_ms}{api_key}{recv_window}{query_string}"
    return hmac.new(API_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _rate_limit_sleep(attempt: int) -> None:
    delay = min(RATE_LIMIT_SLEEP_BASE * attempt * attempt, 30)
    time.sleep(delay)


def _is_rate_limit_response(payload: dict) -> bool:
    return payload.get("retCode") == RATE_LIMIT_CODE


def _safe_get(url, params):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            j = r.json()
            if isinstance(j, dict) and _is_rate_limit_response(j):
                last_err = RuntimeError(
                    f"Rate limited retCode={j.get('retCode')} retMsg={j.get('retMsg')}"
                )
                _rate_limit_sleep(attempt)
                continue
            return j
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * attempt, 6))
    raise RuntimeError(f"HTTP/JSON failed after {MAX_RETRIES} retries: {last_err}")


def _safe_private_get(path: str, params: dict) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            timestamp_ms = str(int(time.time() * 1000))
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = _sign_request(timestamp_ms, API_KEY, RECV_WINDOW, query_string)
            headers = {
                "X-BAPI-API-KEY": API_KEY,
                "X-BAPI-SIGN": signature,
                "X-BAPI-TIMESTAMP": timestamp_ms,
                "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            }
            url = f"{PRIVATE_BASE_URL}{path}"
            r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            j = r.json()
            if isinstance(j, dict) and _is_rate_limit_response(j):
                last_err = RuntimeError(
                    f"Rate limited retCode={j.get('retCode')} retMsg={j.get('retMsg')}"
                )
                _rate_limit_sleep(attempt)
                continue
            return j
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * attempt, 6))
    raise RuntimeError(f"Private HTTP/JSON failed after {MAX_RETRIES} retries: {last_err}")


def _safe_private_post(path: str, payload: dict) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            timestamp_ms = str(int(time.time() * 1000))
            payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            signature = _sign_request(timestamp_ms, API_KEY, RECV_WINDOW, payload_json)
            headers = {
                "X-BAPI-API-KEY": API_KEY,
                "X-BAPI-SIGN": signature,
                "X-BAPI-TIMESTAMP": timestamp_ms,
                "X-BAPI-RECV-WINDOW": RECV_WINDOW,
                "Content-Type": "application/json",
            }
            url = f"{PRIVATE_BASE_URL}{path}"
            r = requests.post(url, data=payload_json, headers=headers, timeout=REQUEST_TIMEOUT)
            j = r.json()
            if isinstance(j, dict) and _is_rate_limit_response(j):
                last_err = RuntimeError(
                    f"Rate limited retCode={j.get('retCode')} retMsg={j.get('retMsg')}"
                )
                _rate_limit_sleep(attempt)
                continue
            return j
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * attempt, 6))
    raise RuntimeError(f"Private HTTP/JSON failed after {MAX_RETRIES} retries: {last_err}")


def _log_nonfatal_order_error(context: str, payload: dict) -> None:
    ret_code = payload.get("retCode")
    ret_msg = payload.get("retMsg")
    print(f"[warn] {context} skipped retCode={ret_code} retMsg={ret_msg} payload={payload}")


def _is_nonfatal_order_error(payload: dict) -> bool:
    return payload.get("retCode") in (110009, 110017)


def load_config_data(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_strategy_config(data: dict, selected_set: Optional[dict] = None) -> StrategyConfig:
    strategy = (selected_set or {}).get("strategy") or data["strategy"]
    short_entry = strategy["short_entry"]
    target_roi_pct = float(
        (selected_set or {}).get("target_roi_pct", data.get("target_roi_pct", TARGET_ROI_PCT_DEFAULT))
    )
    take_profit_pct = TAKE_PROFIT_PCT_DEFAULT  # 0.33% TP — only exit mechanism

    return StrategyConfig(
        symbol=data["symbol"],
        category=data["category"],
        interval=data["interval"],
        position_size=float(data["position_size"]),
        leverage=float(data["leverage"]),
        slippage=float(data["slippage"]),
        short_fast=int(short_entry["fast"]),
        short_mid=int(short_entry["mid"]),
        short_slow=int(short_entry["slow"]),
        short_rsi_len=int(short_entry["rsi_len"]),
        ema_len=int(short_entry.get("ema_len", 50)),
        ma_type=_parse_ma_type(short_entry.get("ma_type", 0)),
        rsi_source=int(short_entry.get("rsi_source", 0)),
        ema_source=int(short_entry.get("ema_source", 0)),
        short_rsi_trig=float(short_entry["rsi_trig"]),
        take_profit_pct=take_profit_pct,
        target_roi_pct=target_roi_pct,
        volume_filter=int(short_entry.get("volume_filter", 0)),
        volume_ma_period=int(short_entry.get("volume_ma_period", 20)),
        volume_threshold=float(short_entry.get("volume_threshold", 1.0)),
        avoid_first_hours=int(short_entry.get("avoid_first_hours", 0)),
        avoid_last_hours=int(short_entry.get("avoid_last_hours", 0)),
    )


def fetch_bybit_ohlcv(symbol: str, category: str, interval: str, limit: int) -> pd.DataFrame:
    params = {
        "category": category,
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    j = _safe_get(f"{PUBLIC_BASE_URL}/v5/market/kline", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")

    rows = (j.get("result") or {}).get("list") or []
    if not rows:
        raise RuntimeError("No candles returned from Bybit.")

    df = pd.DataFrame(rows)
    df = df.iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


def fetch_orderbook_top(symbol: str, category: str) -> dict:
    params = {"category": category, "symbol": symbol.upper(), "limit": 1}
    j = _safe_get(f"{PUBLIC_BASE_URL}/v5/market/orderbook", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Orderbook error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    ob = j.get("result") or {}
    bids = ob.get("b") or []
    asks = ob.get("a") or []
    if not bids or not asks:
        raise RuntimeError("Orderbook empty")
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    return {"best_bid": best_bid, "best_ask": best_ask}


def fetch_instrument_info(symbol: str, category: str) -> dict:
    params = {"category": category, "symbol": symbol.upper()}
    j = _safe_get(f"{PUBLIC_BASE_URL}/v5/market/instruments-info", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Instrument info error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    rows = (j.get("result") or {}).get("list") or []
    if not rows:
        raise RuntimeError(f"No instrument info found for {symbol}")
    return rows[0]


def fetch_fee_rates(symbol: str, category: str) -> dict:
    params = {"category": category, "symbol": symbol.upper()}
    j = _safe_private_get("/v5/account/fee-rate", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Fee-rate error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    rows = (j.get("result") or {}).get("list") or []
    if not rows:
        raise RuntimeError("Fee-rate list empty; cannot determine fees.")
    return rows[0]


def normalize_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    step_dec = Decimal(str(step))
    qty_dec = Decimal(str(qty))
    normalized = (qty_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
    return float(normalized)


def format_qty(qty: float, step: float) -> str:
    if step <= 0:
        return str(qty)
    step_dec = Decimal(str(step))
    qty_dec = Decimal(str(qty))
    return format(qty_dec.quantize(step_dec, rounding=ROUND_DOWN), "f")


def fetch_wallet_balance(coin: str = "USDT") -> float:
    params = {"accountType": "UNIFIED", "coin": coin}
    j = _safe_private_get("/v5/account/wallet-balance", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Wallet balance error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    rows = (j.get("result") or {}).get("list") or []
    if not rows:
        raise RuntimeError("Wallet balance list empty; cannot determine balance.")
    coins = rows[0].get("coin") or []
    for entry in coins:
        if entry.get("coin") == coin:
            return float(entry.get("walletBalance", 0.0))
    raise RuntimeError(f"Coin {coin} not found in wallet balance response.")


def fetch_position(symbol: str, category: str) -> dict:
    params = {"category": category, "symbol": symbol.upper()}
    j = _safe_private_get("/v5/position/list", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Position list error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    positions = (j.get("result") or {}).get("list") or []
    return positions[0] if positions else {}


def fetch_open_orders(symbol: str, category: str) -> list:
    params = {"category": category, "symbol": symbol.upper(), "openOnly": 1}
    j = _safe_private_get("/v5/order/realtime", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Open orders error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    return (j.get("result") or {}).get("list") or []


def fetch_order_status(cfg: StrategyConfig, order_id: str) -> dict:
    params = {"category": cfg.category, "symbol": cfg.symbol.upper(), "orderId": order_id}
    j = _safe_private_get("/v5/order/realtime", params)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Order realtime error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    lst = (j.get("result") or {}).get("list") or []
    if not lst:
        raise RuntimeError(f"Order not found in realtime: {order_id} resp={j}")
    return lst[0]


def close_position_market(cfg: StrategyConfig, position_side: str, qty: float, qty_step: float) -> dict:
    close_side = "Buy" if position_side == "Sell" else "Sell"
    payload = {
        "category": cfg.category,
        "symbol": cfg.symbol,
        "side": close_side,
        "orderType": "Market",
        "qty": format_qty(qty, qty_step),
        "timeInForce": "IOC",
        "reduceOnly": True,
    }
    j = _safe_private_post("/v5/order/create", payload)
    if j.get("retCode") != 0:
        raise RuntimeError(f"Close order create error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")
    order_id = (j.get("result") or {}).get("orderId")
    if not order_id:
        raise RuntimeError(f"Missing orderId in close response: {j}")
    st = fetch_order_status(cfg, order_id)
    return {
        "orderId": order_id,
        "avgPrice": float(st.get("avgPrice", 0) or 0),
        "cumExecQty": float(st.get("cumExecQty", 0) or 0),
    }


def cancel_by_order_id(cfg: StrategyConfig, order_id: str) -> None:
    payload = {"category": cfg.category, "symbol": cfg.symbol, "orderId": order_id}
    j = _safe_private_post("/v5/order/cancel", payload)
    ret_code = j.get("retCode")
    if ret_code == 110001:
        print(f"[warn] cancel skipped: order not found or too late to cancel id={order_id}")
        return
    if ret_code != 0:
        raise RuntimeError(f"Cancel error retCode={ret_code} retMsg={j.get('retMsg')} payload={j}")


def cancel_protective_orders(cfg: StrategyConfig) -> None:
    orders = fetch_open_orders(cfg.symbol, cfg.category)
    for order in orders:
        if order.get("reduceOnly"):
            order_id = order.get("orderId")
            if order_id:
                cancel_by_order_id(cfg, order_id)


def _is_conditional_order(order: dict) -> bool:
    trigger_price = order.get("triggerPrice")
    stop_order_type = order.get("stopOrderType") or ""
    trigger_direction = order.get("triggerDirection")
    has_trigger_price = trigger_price not in (None, "", "0", 0, 0.0)
    has_stop_order_type = stop_order_type not in ("", "UNKNOWN", "None")
    has_trigger_direction = trigger_direction not in (None, 0, "0")
    return has_trigger_price or has_stop_order_type or has_trigger_direction


def cancel_conditional_orders(cfg: StrategyConfig) -> None:
    orders = fetch_open_orders(cfg.symbol, cfg.category)
    for order in orders:
        if _is_conditional_order(order):
            order_id = order.get("orderId")
            if order_id:
                try:
                    cancel_by_order_id(cfg, order_id)
                except Exception as exc:
                    print(f"[warn] conditional cancel failed id={order_id} err={exc}")


def ensure_protective_exits(
    cfg: StrategyConfig,
    position_side: str,
    avg_price: float,
    qty: float,
    qty_step: float,
) -> None:
    if position_side != "Sell":
        print(f"[exits] skip refresh for non-short position side={position_side}")
        return
    take_profit_price = avg_price * (1.0 - cfg.take_profit_pct)
    set_position_tp(cfg, take_profit_price)
    print("[exits] refreshed position TP")


def compute_signals(df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    d = df.copy()

    price_source = get_price_source(d, cfg.rsi_source)
    ema_source = get_price_source(d, cfg.ema_source)

    d["s_fast"] = compute_ma(d["close"], cfg.short_fast, cfg.ma_type)
    d["s_mid"] = compute_ma(d["close"], cfg.short_mid, cfg.ma_type)
    d["s_slow"] = compute_ma(d["close"], cfg.short_slow, cfg.ma_type)
    d["s_rsi"] = RSIIndicator(price_source, cfg.short_rsi_len).rsi()
    d["ema"] = ema_source.ewm(span=cfg.ema_len, adjust=False).mean()

    d["volume_ma"] = d["volume"].rolling(max(cfg.volume_ma_period, 1)).mean()
    d["hour"] = d.index.hour

    for col in ["s_fast", "s_mid", "s_slow", "s_rsi", "ema", "volume_ma", "hour"]:
        d[col] = d[col].shift(1)

    d["ema_trend_up"] = d["ema"] > d["ema"].shift(1)
    volume_ok = True
    if cfg.volume_filter == 1:
        volume_ok = d["volume"].shift(1) > (d["volume_ma"] * cfg.volume_threshold)
    time_ok = True
    if cfg.avoid_first_hours > 0 or cfg.avoid_last_hours > 0:
        hour_prev = d["hour"]
        time_ok = (hour_prev >= cfg.avoid_first_hours) & (hour_prev < (24 - cfg.avoid_last_hours))
    d["short_signal"] = (
        (d["s_fast"] < d["s_mid"])
        & (d["s_mid"] < d["s_slow"])
        & (d["s_rsi"] > cfg.short_rsi_trig)
        & (~d["ema_trend_up"])
        & volume_ok
        & time_ok
    )
    last_closed = d.iloc[-2]
    return {
        "timestamp": d.index[-2],
        "short_signal": bool(last_closed["short_signal"]),
        "last_close": float(last_closed["close"]),
        "last_high": float(last_closed["high"]),
        "last_low": float(last_closed["low"]),
    }


def place_postonly_entry(cfg: StrategyConfig, side: str, qty: float, qty_step: float) -> dict:
    ob = fetch_orderbook_top(cfg.symbol, cfg.category)
    price = ob["best_bid"] if side == "Buy" else ob["best_ask"]

    payload = {
        "category": cfg.category,
        "symbol": cfg.symbol,
        "side": side,
        "orderType": "Limit",
        "qty": format_qty(qty, qty_step),
        "price": str(price),
        "timeInForce": "PostOnly",
    }
    j = _safe_private_post("/v5/order/create", payload)
    if j.get("retCode") != 0:
        if j.get("retCode") == 10001 or "Qty invalid" in str(j.get("retMsg")):
            _log_nonfatal_order_error("entry order create", j)
            return {"filled": False, "orderId": None, "error": j}
        raise RuntimeError(f"Order create error retCode={j.get('retCode')} retMsg={j.get('retMsg')} payload={j}")

    order_id = (j.get("result") or {}).get("orderId")
    if not order_id:
        raise RuntimeError(f"Missing orderId in response: {j}")
    print(f"[entry] post-only order accepted id={order_id} side={side} qty={qty:.6f}")

    deadline = time.time() + 25
    while time.time() < deadline:
        st = fetch_order_status(cfg, order_id)
        status = st.get("orderStatus")
        if status == "Filled":
            return {
                "filled": True,
                "orderId": order_id,
                "avgPrice": float(st.get("avgPrice", 0) or 0),
                "cumExecQty": float(st.get("cumExecQty", 0) or 0),
            }
        if status in ("Cancelled", "Rejected"):
            return {"filled": False, "orderId": order_id}
        time.sleep(0.5)

    cancel_by_order_id(cfg, order_id)
    st = fetch_order_status(cfg, order_id)
    filled_qty = float(st.get("cumExecQty", 0) or 0)
    if filled_qty > 0:
        return {
            "filled": True,
            "orderId": order_id,
            "avgPrice": float(st.get("avgPrice", 0) or 0),
            "cumExecQty": filled_qty,
        }
    return {"filled": False, "orderId": order_id}


def set_position_tp(cfg: StrategyConfig, tp_price: float) -> None:
    payload = {
        "category": cfg.category,
        "symbol": cfg.symbol,
        "takeProfit": str(round(tp_price, 8)),
        "tpTriggerBy": "MarkPrice",
        "tpslMode": "Full",
        "positionIdx": 0,
    }
    j = _safe_private_post("/v5/position/trading-stop", payload)
    if j.get("retCode") != 0:
        print(f"[warn] set_position_tp failed retCode={j.get('retCode')} retMsg={j.get('retMsg')}")
    else:
        print(f"[exit] position TP set at {tp_price:.6f}")


def place_protective_exits(
    cfg: StrategyConfig,
    entry_side: str,
    entry_price: float,
    qty: float,
    qty_step: float = 0.0,
) -> None:
    if entry_side != "Sell":
        raise ValueError(f"Short-only strategy: unsupported entry_side={entry_side}")
    take_profit_price = entry_price * (1.0 - cfg.take_profit_pct)
    set_position_tp(cfg, take_profit_price)



def run_loop():
    config_data = load_config_data()
    cfg = build_strategy_config(config_data)
    max_window = max(
        cfg.short_slow,
        cfg.short_rsi_len,
        cfg.ema_len,
        cfg.volume_ma_period,
    ) + 5
    last_candle_time = None
    last_config_refresh = time.time()
    last_profit_check = 0.0
    start_balance = fetch_wallet_balance("USDT")
    instrument = fetch_instrument_info(cfg.symbol, cfg.category)
    fee_rates = fetch_fee_rates(cfg.symbol, cfg.category)
    taker_fee = float(fee_rates.get("takerFeeRate", TAKER_FEE_DEFAULT))
    maker_fee = float(fee_rates.get("makerFeeRate", MAKER_FEE_DEFAULT))
    lot_filter = instrument.get("lotSizeFilter") or {}
    min_qty = float(lot_filter.get("minOrderQty", 0) or 0)
    qty_step = float(lot_filter.get("qtyStep", 0) or 0)
    last_exit_key = None
    entry_size = None
    print(f"[startup] loaded config for {cfg.symbol} {cfg.interval}m ({cfg.category})")

    while True:
        loop_start = time.time()
        if time.time() - last_config_refresh >= CONFIG_REFRESH_SECONDS:
            config_data = load_config_data()
            cfg = build_strategy_config(config_data)
            max_window = max(
                cfg.short_slow,
                cfg.short_rsi_len,
                cfg.ema_len,
                cfg.volume_ma_period,
            ) + 5
            last_config_refresh = time.time()
            instrument = fetch_instrument_info(cfg.symbol, cfg.category)
            fee_rates = fetch_fee_rates(cfg.symbol, cfg.category)
            taker_fee = float(fee_rates.get("takerFeeRate", TAKER_FEE_DEFAULT))
            maker_fee = float(fee_rates.get("makerFeeRate", MAKER_FEE_DEFAULT))
            lot_filter = instrument.get("lotSizeFilter") or {}
            min_qty = float(lot_filter.get("minOrderQty", 0) or 0)
            qty_step = float(lot_filter.get("qtyStep", 0) or 0)
            print(f"[config] reloaded config for {cfg.symbol} {cfg.interval}m ({cfg.category})")

        if time.time() - last_profit_check >= PROFIT_CHECK_INTERVAL:
            last_profit_check = time.time()
            balance = fetch_wallet_balance("USDT")
            if balance >= start_balance * PROFIT_PAUSE_THRESHOLD:
                interval_minutes = max(int(float(cfg.interval)), 1)
                pause_seconds = PROFIT_PAUSE_CANDLES * interval_minutes * 60
                print(
                    f"[profit] balance={balance:.4f} reached {PROFIT_PAUSE_THRESHOLD:.2f}x "
                    f"start={start_balance:.4f}; pausing for {PROFIT_PAUSE_CANDLES} candles ("
                    f"~{pause_seconds}s)"
                )
                time.sleep(pause_seconds)
                _restart_script()

        df = fetch_bybit_ohlcv(cfg.symbol, cfg.category, cfg.interval, max(API_LIMIT, max_window))
        signal = compute_signals(df, cfg)
        if last_candle_time and signal["timestamp"] < last_candle_time:
            print(
                f"[warn] stale candle received ts={signal['timestamp']} last_seen={last_candle_time}; skipping"
            )
            time.sleep(5)
            continue
        if last_candle_time == signal["timestamp"]:
            print(f"[idle] waiting for new candle at {signal['timestamp']}")
            time.sleep(5)
            continue

        last_candle_time = signal["timestamp"]
        position = fetch_position(cfg.symbol, cfg.category)
        size = float(position.get("size", 0.0)) if position else 0.0
        side = position.get("side") if position else None
        avg_price = float(position.get("avgPrice", 0.0)) if position else 0.0
        if size == 0.0:
            entry_size = None
        print(
            f"[signal] ts={signal['timestamp']} close={signal['last_close']:.4f} "
            f"short={signal['short_signal']} pos_size={size}"
        )

        if size == 0.0:
            balance = fetch_wallet_balance("USDT")
            ob = fetch_orderbook_top(cfg.symbol, cfg.category)
            mid = (ob["best_bid"] + ob["best_ask"]) / 2.0
            raw_qty = (balance * cfg.position_size * cfg.leverage) / mid
            qty = normalize_qty(raw_qty, qty_step)
            print(
                f"[sizing] balance={balance:.4f} mid={mid:.4f} raw_qty={raw_qty:.6f} "
                f"qty={qty:.6f} min={min_qty} step={qty_step}"
            )
            if qty <= 0 or (min_qty and qty < min_qty):
                print("[sizing] qty below minimum; skipping entry")
                time.sleep(5)
                continue

            if signal["short_signal"]:
                entry = place_postonly_entry(cfg, "Sell", qty, qty_step)
                if entry.get("filled"):
                    avg_price = float(entry.get("avgPrice") or mid)
                    filled_qty = float(entry.get("cumExecQty") or qty)
                    print(f"[entry] short filled avg={avg_price:.4f} qty={filled_qty:.6f}")
                    place_protective_exits(
                        cfg,
                        "Sell",
                        avg_price,
                        filled_qty,
                        qty_step=qty_step,
                    )
                    print("[exits] placed protective TP for short")
                    last_exit_key = ("Sell", filled_qty, avg_price)
                    entry_size = filled_qty
                else:
                    print("[entry] short not filled (skipped)")
            else:
                print("[signal] no entry conditions met")
        else:
            exit_key = (side, size, avg_price)
            if avg_price > 0 and side in ("Buy", "Sell") and exit_key != last_exit_key:
                print(f"[position] existing position detected side={side} size={size} avg={avg_price:.4f}")
                if side == "Sell":
                    ensure_protective_exits(cfg, side, avg_price, size, qty_step)
                else:
                    print(f"[exits] skip refresh for non-short position side={side}")
                last_exit_key = exit_key
            else:
                print("[position] in position; skipping new entries")

        elapsed = time.time() - loop_start
        print(f"[loop] cycle complete in {elapsed:.2f}s")
        time.sleep(5)


if __name__ == "__main__":
    run_loop()
"""


def _module_from_code(name: str, code: str):
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(code, f"<{name}>", "exec"), module.__dict__)
    return module


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.json not found; run backtest first.")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _run_backtest_once(backtester) -> None:
    print("[orchestrator] starting backtest cycle...")
    backtester.run_backtest_cycle()
    if not CONFIG_PATH.exists():
        print(
            "[orchestrator] no config generated; backtest did not evaluate any configurations."
        )
    if not CONFIG_PATH.exists():
        print(
            "[orchestrator] backtest produced no config; writing fallback config from history."
        )
        history_path = Path(
            getattr(backtester, "OPTIMIZER_HISTORY_PATH", "optimizer_history.jsonl")
        )
        best_record = None
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    metrics = record.get("metrics", {})
                    mean_return = float(metrics.get("mean_return", -1e18))
                    score = (mean_return,)
                    if best_record is None:
                        best_record = record
                        best_record["_combo_score"] = score
                        continue
                    if score > best_record.get("_combo_score", (-1e18,)):
                        best_record = record
                        best_record["_combo_score"] = score
        if best_record:
            metrics = best_record.get("metrics", {})
            print(
                "[orchestrator] best recorded metrics="
                f"{metrics} combo_score={best_record.get('_combo_score')}"
            )
            params = backtester._normalize_params(best_record["params"])
        else:
            params = backtester._normalize_params(
                backtester.PARAM_SPACE.sample_random(
                    backtester.np.random.default_rng(backtester.OPTIMIZER_SEED)
                )
            )
        (
            s_fast,
            s_mid,
            s_slow,
            s_rsi_len,
            ema_len,
            ma_type,
            rsi_source,
            ema_source,
            volume_filter,
            volume_ma_period,
            volume_threshold,
            avoid_first_hours,
            avoid_last_hours,
            s_rsi_trig,
            target_roi_pct,
        ) = params
        ma_type_label = {0: "SMA", 1: "EMA", 2: "WMA"}.get(int(ma_type), "SMA")
        leverage = 1.0
        if hasattr(backtester, "fetch_position_leverage"):
            try:
                leverage = float(backtester.fetch_position_leverage(backtester.SYMBOL))
            except Exception:
                leverage = 1.0
        config = {
            "symbol": backtester.SYMBOL,
            "category": backtester.CATEGORY,
            "interval": backtester.INTERVAL,
            "days": getattr(backtester, "DAYS", 0),
            "balance": backtester.START_BALANCE,
            "position_size": backtester.POSITION_SIZE,
            "leverage": leverage,
            "fees": {
                "taker_fee": backtester.TAKER_FEE,
                "maker_fee": 0.0,
            },
            "slippage": backtester.SLIPPAGE,
            "target_roi_pct": target_roi_pct,
            "take_profit_pct": backtester.TAKE_PROFIT_PCT,
            "strategy": {
                "short_entry": {
                    "fast": s_fast,
                    "mid": s_mid,
                    "slow": s_slow,
                    "rsi_len": s_rsi_len,
                    "ema_len": ema_len,
                    "ma_type": ma_type_label,
                    "rsi_source": rsi_source,
                    "ema_source": ema_source,
                    "rsi_trig": s_rsi_trig,
                    "volume_filter": volume_filter,
                    "volume_ma_period": volume_ma_period,
                    "volume_threshold": volume_threshold,
                    "avoid_first_hours": avoid_first_hours,
                    "avoid_last_hours": avoid_last_hours,
                },
                "short_exit": {},
            },
            "parameter_sets": [
                {
                    "name": "set_1",
                    "strategy": {
                        "short_entry": {
                            "fast": s_fast,
                            "mid": s_mid,
                            "slow": s_slow,
                            "rsi_len": s_rsi_len,
                            "ema_len": ema_len,
                            "ma_type": ma_type_label,
                            "rsi_source": rsi_source,
                            "ema_source": ema_source,
                            "rsi_trig": s_rsi_trig,
                            "volume_filter": volume_filter,
                            "volume_ma_period": volume_ma_period,
                            "volume_threshold": volume_threshold,
                            "avoid_first_hours": avoid_first_hours,
                            "avoid_last_hours": avoid_last_hours,
                        },
                        "short_exit": {},
                    },
                    "target_roi_pct": target_roi_pct,
                }
            ],
            "regime_model": {
                "feature_order": [
                    "volatility",
                    "trend_strength",
                    "volume_z",
                    "return_skew",
                    "return_kurt",
                ],
                "centers": [[0.0] * 5, [0.0] * 5],
                "trade_cluster": 0,
                "window": backtester.REGIME_WINDOW,
            },
            "regime_selector": {
                "feature_order": ["volatility", "trend_strength"],
                "tree": {
                    "feature": "volatility",
                    "threshold": 0.0,
                    "left": {"set": "set_1"},
                    "right": {"set": "set_1"},
                },
            },
            "anomaly_model": {
                "feature_order": [
                    "volatility",
                    "trend_strength",
                    "volume_z",
                    "return_skew",
                    "return_kurt",
                ],
                "mean": [0.0] * 5,
                "std": [1.0] * 5,
                "z_thresh": backtester.ANOMALY_Z_THRESH,
                "window": backtester.REGIME_WINDOW,
            },
        }
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
    if not CONFIG_PATH.exists():
        raise RuntimeError("Backtest completed but config.json was not created.")
    print("[orchestrator] backtest complete; config.json saved.")


def _backtest_scheduler(backtester) -> None:
    while True:
        print(
            f"[orchestrator] sleeping {BACKTEST_INTERVAL_SECONDS}s until next backtest"
        )
        time.sleep(BACKTEST_INTERVAL_SECONDS)
        try:
            _run_backtest_once(backtester)
        except Exception as exc:
            print(f"[orchestrator] backtest cycle failed: {exc}")


def _run_ml_loop(live_module, ml_module) -> None:
    print("[ml] starting ML monitoring loop...")
    last_config_refresh = 0.0
    config: Dict[str, Any] = {}
    regime_detector: Optional[Any] = None
    anomaly_detector: Optional[Any] = None
    poll_seconds = 60

    while True:
        now = time.time()
        if now - last_config_refresh >= 5 * 60 or not config:
            config = _load_config()
            regime_detector = ml_module.RegimeDetector(
                (config.get("regime_model") or {}).get("feature_order")
                or [
                    "volatility",
                    "trend_strength",
                    "volume_z",
                    "return_skew",
                    "return_kurt",
                ],
                ml_module.np.array(
                    (config.get("regime_model") or {}).get("centers")
                    or ml_module.np.zeros((2, 5)),
                    dtype=float,
                ),
                int((config.get("regime_model") or {}).get("trade_cluster", 0)),
            )
            anomaly_detector = ml_module.AnomalyDetector(
                (config.get("anomaly_model") or {}).get("feature_order")
                or [
                    "volatility",
                    "trend_strength",
                    "volume_z",
                    "return_skew",
                    "return_kurt",
                ],
                ml_module.np.array(
                    (config.get("anomaly_model") or {}).get("mean")
                    or ml_module.np.zeros(5),
                    dtype=float,
                ),
                ml_module.np.array(
                    (config.get("anomaly_model") or {}).get("std")
                    or ml_module.np.ones(5),
                    dtype=float,
                ),
                float((config.get("anomaly_model") or {}).get("z_thresh", 3.5)),
            )
            interval_minutes = int(config.get("interval", 1))
            poll_seconds = max(60, interval_minutes * 60)
            last_config_refresh = now
            print("[ml] config reloaded for ML loop")

        if regime_detector is None or anomaly_detector is None:
            time.sleep(5)
            continue

        df = live_module.fetch_bybit_ohlcv(
            config["symbol"],
            config["category"],
            config["interval"],
            max(
                live_module.API_LIMIT,
                int((config.get("regime_model") or {}).get("window", 120)) + 5,
            ),
        )
        features = ml_module.compute_regime_features(
            df,
            int((config.get("regime_model") or {}).get("window", 120)),
        )
        ema_len = int(
            ((config.get("strategy") or {}).get("short_entry") or {}).get("ema_len", 50)
        )
        ema = df["close"].ewm(span=ema_len, adjust=False).mean()
        ema_trend_up = bool(ema.iloc[-1] > ema.iloc[-2]) if len(ema) > 1 else False
        regime = regime_detector.classify(features)
        is_anomaly = anomaly_detector.is_anomaly(features)
        print(
            f"[ml] regime={regime} anomaly={is_anomaly} ema_trend_up={ema_trend_up} features={features}"
        )
        time.sleep(poll_seconds)


def main() -> None:
    ml_module = _module_from_code("ml_layers", ML_LAYERS_CODE)
    sys.modules["ml_layers"] = ml_module

    backtester = _module_from_code("backtester", BACKTESTER_CODE)
    live_module = _module_from_code("live_trading", LIVE_CODE)

    if API_KEY:
        backtester.API_KEY = API_KEY
        live_module.API_KEY = API_KEY
    if API_SECRET:
        backtester.API_SECRET = API_SECRET
        live_module.API_SECRET = API_SECRET

    _run_backtest_once(backtester)

    live_thread = threading.Thread(
        target=live_module.run_loop, name="live-trading", daemon=True
    )
    ml_thread = threading.Thread(
        target=_run_ml_loop, args=(live_module, ml_module), name="ml-loop", daemon=True
    )
    backtest_thread = threading.Thread(
        target=_backtest_scheduler,
        args=(backtester,),
        name="backtest-scheduler",
        daemon=True,
    )

    live_thread.start()
    ml_thread.start()
    backtest_thread.start()

    print("[orchestrator] live trading + ML running. Backtest scheduler active.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[orchestrator] shutdown requested")


if __name__ == "__main__":
    main()
