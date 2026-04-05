# BACK2BASICS — Short Trading System
### A White Paper

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Signal Generation & Entry Logic](#4-signal-generation--entry-logic)
5. [Exit Mechanics](#5-exit-mechanics)
6. [Risk Management Framework](#6-risk-management-framework)
7. [Bayesian Parameter Optimization](#7-bayesian-parameter-optimization)
8. [Walk-Forward Validation](#8-walk-forward-validation)
9. [Machine Learning Layer](#9-machine-learning-layer)
10. [Live Execution Engine](#10-live-execution-engine)
11. [Exchange Integration (Bybit API)](#11-exchange-integration-bybit-api)
12. [Configuration Reference](#12-configuration-reference)
13. [Installation & Setup](#13-installation--setup)
14. [Operational Workflow](#14-operational-workflow)
15. [Performance Metrics](#15-performance-metrics)
16. [Limitations & Disclaimers](#16-limitations--disclaimers)

---

## 1. Abstract

BACK2BASICS is a self-optimizing, short-only algorithmic trading system designed to operate on the Bybit perpetual futures exchange. Built around the philosophy of simplicity and robustness, the system combines classical technical analysis — moving average alignment, RSI momentum, and EMA trend filtering — with modern machine learning and Bayesian hyperparameter optimization.

The system operates in three concurrent threads: a **backtesting/optimization engine** that re-evaluates strategy parameters every 8 hours using walk-forward validation; a **live execution engine** that manages real-time order placement and position lifecycle on Bybit; and a **machine learning monitor** that classifies market regime and detects anomalies to gate or suppress trading activity.

The core principle is disciplined short-selling: the system seeks to enter positions in confirmed downtrends and exit quickly at predefined take-profit targets. It avoids stop-losses, instead relying on high-confidence entries, small position sizing, and adaptive parameter tuning to manage risk.

---

## 2. System Overview

| Property | Value |
|---|---|
| Exchange | Bybit (linear perpetual futures) |
| Default Symbol | ESPUSDT |
| Strategy Direction | Short-only |
| Default Timeframe | 15-minute candles (adaptive) |
| Optimization Cycle | Every 8 hours |
| Position Sizing | 60% of equity |
| Default Leverage | Fetched live from exchange |
| Target Take-Profit | 0.33% per trade |
| Entry Order Type | Post-only limit (maker) |
| Language | Python 3.x |

The system runs as a single Python script (`BACK2BASICS-SHORTTRADER.py`) that self-contains three fully functional sub-modules as embedded strings, which are compiled and loaded at runtime using Python's `types.ModuleType` and `compile()` machinery. This design means the entire system — backtester, live trader, and ML layer — is distributed as one file with zero external module dependencies beyond standard scientific Python packages.

---

## 3. Architecture

The system is structured as an **orchestrator** with three daemon threads and one embedded startup backtest.

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (main.py)                    │
│                                                             │
│  ┌──────────────────┐  ┌────────────────┐  ┌────────────┐  │
│  │  Backtest Thread │  │  Live Thread   │  │  ML Thread │  │
│  │  (8h scheduler)  │  │  (5s loop)     │  │  (60s loop)│  │
│  └──────────────────┘  └────────────────┘  └────────────┘  │
│           │                    │                  │          │
│           └───────config.json──┘                 │          │
│                    │                             │          │
│             Bybit REST API              Bybit REST API      │
└─────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | Role |
|---|---|
| `ML_LAYERS_CODE` | Regime detection, anomaly detection, feature engineering |
| `BACKTESTER_CODE` | Historical OHLCV fetching, signal backtesting, TPE optimization |
| `LIVE_CODE` | Real-time candle fetching, signal computation, order management |

All three modules share API credentials via the orchestrator, which injects `API_KEY` and `API_SECRET` into each module's namespace at startup.

---

## 4. Signal Generation & Entry Logic

The strategy is **short-only** and requires all of the following conditions to align on the same bar before entering a position:

### 4.1 Moving Average Cascade

Three moving averages of configurable type and period must be in **descending alignment**:

```
fast_MA < mid_MA < slow_MA
```

This cascade structure confirms a sustained downtrend across multiple time horizons — the fast line below the mid line signals near-term bearishness, while the mid below slow confirms the broader trend is also declining.

**MA Type** is configurable and selected during optimization:

| Code | Type |
|---|---|
| 0 | Simple Moving Average (SMA) |
| 1 | Exponential Moving Average (EMA) |
| 2 | Weighted Moving Average (WMA) |

**Period Hierarchy Enforcement:** The optimizer enforces strict ordering — `s_mid >= s_fast + 2` and `s_slow >= s_mid + 5` — so the cascade always uses genuinely distinct time horizons.

### 4.2 RSI Momentum Confirmation

The RSI must be **above** a configured trigger threshold (`s_rsi_trig`). This is a non-standard RSI application: rather than using RSI overbought/oversold levels to call reversals, this system uses RSI above a threshold as **momentum confirmation** of continued downside pressure, consistent with the MA cascade.

**RSI Source** is configurable:

| Code | Price Source |
|---|---|
| 0 | Close |
| 1 | HL2 (High + Low) / 2 |
| 2 | HLC3 (High + Low + Close) / 3 |

### 4.3 EMA Trend Filter

An independent EMA of configurable length (`ema_len`) must be in **downtrend** at the time of entry:

```
ema[-1] < ema[-2]  →  ema_trend_up = False  →  allow short
```

If the EMA is rising, no short entries are permitted regardless of MA cascade or RSI. This filter acts as a final macro-trend gate.

**EMA Source** is also configurable (close, hl2, hlc3), allowing optimization of the most responsive price input for this filter.

### 4.4 Volume Filter (Optional)

When enabled (`volume_filter = 1`), the bar's volume must exceed a dynamic volume threshold:

```
volume > volume_MA(volume_ma_period) * volume_threshold
```

This ensures entries occur only during active market conditions, avoiding thin liquidity periods where slippage and false signals are more likely.

### 4.5 Time-of-Day Filter (Optional)

The strategy can be configured to avoid trading during the first `avoid_first_hours` and last `avoid_last_hours` of the UTC trading day. This sidesteps periods of historically lower liquidity or distorted price action around daily open/close transitions.

### 4.6 No-Lookahead Guarantee

All indicators are computed using data shifted backward by one bar before signal evaluation:

```python
d[col] = d[col].shift(1)
```

This means the signal on bar `i` is evaluated using indicators from bar `i-1`, and the trade executes at bar `i`'s open price. This is a strict no-lookahead, no-repaint guarantee that prevents backtest overfitting from future data contamination.

---

## 5. Exit Mechanics

### 5.1 Take-Profit Target

The system uses a single, fixed exit mechanism: a **take-profit limit** placed at:

```
take_profit_price = entry_price * (1 - TAKE_PROFIT_PCT)
```

In the backtest, the exit is triggered when a bar's **low** drops to or below the take-profit price, simulating a realistic fill. In live trading, this is submitted as a take-profit order to Bybit's position management API (`/v5/position/trading-stop`), executed server-side.

### 5.2 No Stop-Loss Design

BACK2BASICS deliberately omits a traditional stop-loss. The rationale:

- Small position size (60% of equity, no leverage amplification beyond what the exchange provides) limits absolute loss per trade.
- High-confidence entry conditions (5 filters must all align) result in a naturally high signal-to-noise ratio.
- The optimizer penalizes parameter sets with high maximum drawdown, indirectly selecting for configurations where losing trades are modest.
- A stop-loss would introduce an additional optimization parameter that can be overfitted; avoiding it reduces the dimensionality of the parameter space.

Traders deploying this system should be aware of this design choice and consider their own risk tolerance independently.

### 5.3 Profit Pause Mechanism

The live module monitors cumulative performance. If the running balance reaches **1.5x the starting balance**, trading is paused for 400 candles (approximately 100 hours at the 15-minute default timeframe). After the pause, the process restarts via `os.execv()`. This mechanism prevents runaway compounding during hot streaks that may be unsustainable.

---

## 6. Risk Management Framework

### 6.1 Position Sizing

```
qty = (balance * POSITION_SIZE * leverage) / entry_price
```

With `POSITION_SIZE = 0.60`, no more than 60% of available equity is deployed in any single trade. This leaves a capital buffer for drawdown tolerance and avoids liquidation risk on moderate adverse moves.

### 6.2 Fee Modeling

The backtester models all costs realistically:

| Cost Type | Rate | Notes |
|---|---|---|
| Taker fee | 0.075% (default) | Overridden by live fetch from Bybit |
| Maker fee | 0.0% (default) | Post-only entries qualify for rebate |
| Slippage | 0.01% | Applied to exit fills |

Maker fee is fetched live from the Bybit fee-rate API before each optimization cycle, ensuring the backtest reflects the account's actual fee tier.

### 6.3 Equity Floor

If simulated equity falls to or below $0.50 USDT during a backtest, the simulation terminates. This prevents degenerate parameter combinations from producing meaningless negative-equity statistics.

### 6.4 Drawdown Tracking

The backtester computes **maximum drawdown** (peak-to-trough equity decline) for every backtest run. This metric is reported in the summary and factored into the cross-validation scoring when selecting optimal parameters.

---

## 7. Bayesian Parameter Optimization

Rather than performing an exhaustive grid search across all parameter combinations (which would be computationally prohibitive with 15 parameters), BACK2BASICS implements a **Tree-Parzen Estimator (TPE)** — a form of Bayesian optimization — to efficiently search the parameter space.

### 7.1 Parameter Space

The following 15 parameters are tuned during each optimization cycle:

| Parameter | Type | Description |
|---|---|---|
| `s_fast` | int | Fast MA period |
| `s_mid` | int | Mid MA period (>= s_fast + 2) |
| `s_slow` | int | Slow MA period (>= s_mid + 5) |
| `s_rsi_len` | int | RSI lookback length |
| `ema_len` | int | Trend filter EMA length |
| `ma_type` | int | MA type (0=SMA, 1=EMA, 2=WMA) |
| `rsi_source` | int | RSI price source |
| `ema_source` | int | EMA price source |
| `volume_filter` | int | Volume filter on/off |
| `volume_ma_period` | int | Volume MA window |
| `volume_threshold` | float | Volume multiplier threshold |
| `avoid_first_hours` | int | Hours to skip at day start |
| `avoid_last_hours` | int | Hours to skip at day end |
| `s_rsi_trig` | float | RSI entry trigger level |
| `target_roi_pct` | float | Target ROI weighting |

### 7.2 TPE Algorithm

The TPE divides historical parameter trials into two buckets based on a configurable **gamma** quantile (default: top 20% = "good", bottom 80% = "bad"):

1. **Model fitting**: For each parameter, fit a Gaussian distribution over the "good" and "bad" sets independently.
2. **Candidate sampling**: Generate N candidate parameter vectors by sampling from the "good" Gaussian.
3. **Likelihood ratio**: Score each candidate by the ratio `p(good) / p(bad)` — the Expected Improvement proxy.
4. **Selection**: Return the candidate with the highest likelihood ratio.

This approach concentrates exploration in regions of the parameter space that historically produced high returns, while still maintaining an `explore_ratio` (15%) of pure random samples to avoid premature convergence.

### 7.3 Optimization History Persistence

Results are persisted to per-interval JSONL files (`optimizer_history_15m.jsonl`, etc.). On subsequent optimization cycles, the optimizer loads this history and continues from where it left off, accumulating knowledge across multiple 8-hour cycles rather than restarting blind.

### 7.4 Parallel Evaluation

Parameter combinations are evaluated in parallel using Python's `multiprocessing` pool with `N_WORKERS = cpu_count - 1` processes. Batches of `N_WORKERS * 4` combinations are dispatched at a time, with a 120-second per-batch timeout to guard against hangs.

---

## 8. Walk-Forward Validation

To reduce overfitting risk, the optimization does not score parameters against the full historical window. Instead, it uses **k-fold cross-validation**:

1. The historical OHLCV DataFrame is split into `OPTIMIZER_FOLDS = 3` equal time slices.
2. Each parameter combination is evaluated independently on each slice.
3. Metrics are **aggregated across folds** using conservative statistics:
   - Mean return (optimization target)
   - Standard deviation of returns (consistency indicator)
   - **Worst** drawdown across all folds (not average)
   - **Minimum** profit factor across all folds (not average)
   - Sum of trade counts across all folds

Using worst-case (not average) drawdown and profit factor as reported metrics means the system surfaces configurations that are robust across all time periods, not just lucky on one slice.

### 8.1 Aggregate Metrics Reported

| Metric | Aggregation |
|---|---|
| `mean_return` | Mean across folds |
| `std_return` | Std dev across folds |
| `worst_dd` | Min (worst) across folds |
| `mean_win_rate` | Mean across folds |
| `total_trades` | Sum across folds |
| `profit_factor` | Min (worst) across folds |
| `expectancy` | Mean across folds |
| `net_profit` | Mean across folds |

### 8.2 Multi-Interval Search

The optimizer runs the full TPE cycle for each configured interval in `INTERVALS`. After all intervals are evaluated, the system selects the **interval and parameter set that produced the highest mean return**, then writes that to `config.json` and updates the live trading module.

---

## 9. Machine Learning Layer

The ML layer runs on its own thread, polling every `interval_minutes` seconds. It provides market context that can influence trading decisions but does not directly place or cancel orders in the current implementation — it surfaces its classification for observability and future integration.

### 9.1 Feature Engineering

Five statistical features are extracted from a rolling window (default: 120 bars):

| Feature | Formula |
|---|---|
| `volatility` | Standard deviation of log returns over window |
| `trend_strength` | Linear regression slope / mean close price |
| `volume_z` | (current bar volume − mean volume) / std volume |
| `return_skew` | Skewness of log returns over window |
| `return_kurt` | Excess kurtosis of log returns over window |

### 9.2 RegimeDetector

The `RegimeDetector` classifies the current market regime using **unsupervised k-means clustering** trained on historical feature vectors.

**Training (Fit):**
- Runs k-means (default k=2) over all available feature history.
- Identifies the "trade cluster" as the cluster with the most negative `trend_strength` (i.e., strongest downtrend character).
- Tie-breaking: if volatility is below median, prefer the cluster with lower volatility; otherwise prefer highest-volatility cluster.

**Inference:**
- Computes the Euclidean distance from the current feature vector to each cluster centroid.
- Returns `"TRADE"` if the nearest cluster matches `trade_cluster`, else `"NO_TRADE"`.

**Cluster center persistence:** Fitted cluster centers are serialized to `config.json` under `regime_model.centers`, allowing the live ML thread to reload them without refitting.

### 9.3 AnomalyDetector

The `AnomalyDetector` uses a simple **Z-score based univariate outlier model**:

**Training (Fit):**
- Computes mean and standard deviation for each of the 5 features over historical data.

**Inference:**
- For the current feature vector, computes Z-scores: `z_i = (x_i - mean_i) / std_i`
- Flags the bar as an anomaly if **any** `|z_i| > z_thresh` (default threshold: 3.5)

An anomaly flag indicates the current market conditions are statistically unusual — e.g., a volatility spike, flash crash, or extreme volume — suggesting the signal model may be unreliable.

### 9.4 ML Monitoring Loop

Every polling cycle, the ML thread:

1. Reloads `config.json` (every 5 minutes).
2. Fetches latest candles from Bybit.
3. Computes regime features.
4. Checks EMA trend direction independently.
5. Classifies regime and checks for anomalies.
6. Logs: `[ml] regime=TRADE anomaly=False ema_trend_up=False features={...}`

The regime and anomaly outputs are currently informational and printed to stdout. Integration with the live trading gate is a natural next development step.

---

## 10. Live Execution Engine

### 10.1 Event Loop

The live trading thread runs on a **5-second cycle**:

```
while True:
    ├── Config reload (every 4 hours)
    ├── Fetch candles (API_LIMIT = 200 bars)
    ├── Compute signals (shifted indicators)
    ├── Query current position (/v5/position/list)
    │
    ├── [If flat]
    │   ├── If short_signal: place post-only sell order
    │   ├── Wait 25 seconds for fill confirmation
    │   └── If filled: set take-profit via trading-stop API
    │
    └── [If in position]
        └── Verify take-profit is still active; refresh if needed
```

### 10.2 Post-Only Entry

Entries are placed as **post-only limit orders** at the current best bid price. Post-only orders are guaranteed to rest on the order book and receive maker fee treatment. If the order is not filled within 25 seconds (i.e., price moved away), it is automatically cancelled and the system waits for the next signal.

### 10.3 Take-Profit Submission

Once a short entry is confirmed filled, the system calls Bybit's `/v5/position/trading-stop` endpoint to set a server-side take-profit. This means the take-profit will trigger even if the client process is temporarily disconnected, adding resilience to network interruptions.

### 10.4 Config Hot-Reload

The live module reloads `config.json` every 4 hours. Since the backtest thread rewrites `config.json` every 8 hours with freshly optimized parameters, the live module will automatically adopt the new strategy configuration at the next reload boundary — no restart required.

---

## 11. Exchange Integration (Bybit API)

### 11.1 Authentication

All private API calls use **HMAC-SHA256 request signing** with a timestamp and recv window. Credentials are loaded from environment variables (`BYBIT_API_KEY`, `BYBIT_API_SECRET`) via `.env` file using `python-dotenv`.

### 11.2 API Endpoints Used

**Public (unauthenticated):**

| Endpoint | Purpose |
|---|---|
| `GET /v5/market/kline` | Fetch OHLCV history (paginated) |
| `GET /v5/market/orderbook` | Fetch best bid/ask spread |
| `GET /v5/market/instruments-info` | Lot size, price tick, quantity precision |

**Private (signed):**

| Endpoint | Purpose |
|---|---|
| `GET /v5/account/wallet-balance` | USDT available balance |
| `GET /v5/account/fee-rate` | Account-specific taker/maker rates |
| `GET /v5/position/list` | Current open position |
| `POST /v5/order/create` | Place limit order |
| `POST /v5/order/cancel` | Cancel unfilled order |
| `GET /v5/order/realtime` | Check order fill status |
| `POST /v5/position/trading-stop` | Set take-profit on open position |

### 11.3 Rate Limit Handling

The HTTP client implements **exponential backoff** on rate limit errors (Bybit retCode `10006`):

```
wait = (attempt^2) * 2 seconds
attempt 1 → 2s, attempt 2 → 8s, attempt 3 → 18s, ...
max 6 retries
```

A 12-second request timeout guards against hung connections.

### 11.4 OHLCV Pagination

Historical data is fetched in reverse chronological order using Bybit's `end` parameter pagination, accumulating up to `optimizer_days` of history. This handles Bybit's per-request candle limit gracefully for longer optimization windows.

---

## 12. Configuration Reference

The `config.json` file written by the backtester and read by the live module contains:

```json
{
  "symbol": "ESPUSDT",
  "category": "linear",
  "interval": "15",
  "strategy": {
    "short_entry": {
      "s_fast": 8,
      "s_mid": 21,
      "s_slow": 55,
      "s_rsi_len": 14,
      "ema_len": 50,
      "ma_type": 1,
      "rsi_source": 0,
      "ema_source": 0,
      "volume_filter": 1,
      "volume_ma_period": 20,
      "volume_threshold": 1.2,
      "avoid_first_hours": 1,
      "avoid_last_hours": 1,
      "s_rsi_trig": 55.0,
      "target_roi_pct": 4.0
    }
  },
  "regime_model": {
    "feature_order": ["volatility", "trend_strength", "volume_z", "return_skew", "return_kurt"],
    "centers": [[...], [...]],
    "trade_cluster": 0,
    "window": 120
  },
  "anomaly_model": {
    "feature_order": ["volatility", "trend_strength", "volume_z", "return_skew", "return_kurt"],
    "mean": [...],
    "std": [...],
    "z_thresh": 3.5
  }
}
```

---

## 13. Installation & Setup

### 13.1 Prerequisites

```
Python 3.9+
numpy
pandas
scipy (for ML layer)
python-dotenv
requests
plotext (optional — enables graphical equity curve in terminal)
```

Install dependencies:

```bash
pip install numpy pandas scipy python-dotenv requests plotext
```

### 13.2 Environment Variables

Create a `.env` file in the same directory as the script:

```env
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
```

The script also checks `../1/.env` as a fallback path for multi-strategy setups.

### 13.3 Running

```bash
python BACK2BASICS-SHORTTRADER.py
```

On startup, the orchestrator will:

1. Inject API credentials into sub-modules.
2. Run one full backtest/optimization cycle immediately (blocking).
3. Write `config.json` with the best parameters found.
4. Start the live trading thread, ML thread, and 8-hour backtest scheduler.

---

## 14. Operational Workflow

```
Startup
  │
  ▼
Run backtest cycle (blocking)
  ├── Fetch OHLCV for each interval
  ├── Run TPE optimization (N_COMBOS_PER_INTERVAL = 1,000 trials per interval)
  ├── Score with walk-forward cross-validation
  ├── Select best interval + params
  ├── Print summary to terminal
  └── Write config.json

  ▼
Start 3 daemon threads:
  │
  ├── Live Trading Thread (5-second loop)
  │   ├── Reload config (every 4h)
  │   ├── Compute signals
  │   ├── Place/monitor orders
  │   └── Check profit pause
  │
  ├── ML Monitor Thread (60s loop)
  │   ├── Reload config (every 5 min)
  │   ├── Fetch candles
  │   ├── Compute regime + anomaly
  │   └── Log classification
  │
  └── Backtest Scheduler Thread (every 8h)
      ├── Wait 8 hours
      └── Re-run full optimization cycle
          └── Overwrite config.json with new best params

  ▼
Main thread sleeps (1s loop), handles KeyboardInterrupt
```

---

## 15. Performance Metrics

The following metrics are computed and reported at the end of each optimization cycle:

| Metric | Description |
|---|---|
| `return_pct` | Total percentage return over backtest window |
| `net_profit` | Absolute USDT profit |
| `final_equity` | Ending USDT balance |
| `trades` | Total number of completed trades |
| `wins` | Number of profitable trades |
| `win_rate` | wins / trades |
| `max_dd` | Maximum peak-to-trough equity drawdown (%) |
| `max_loss` | Largest single-trade loss in USDT |
| `max_loss_pct` | Largest single-trade loss as % of equity |
| `profit_factor` | Gross profit / gross loss |
| `expectancy` | Average expected profit per trade in USDT |
| `score (edge)` | Mean return across cross-validation folds |

The equity curve is rendered in the terminal as a Unicode sparkline (▁▂▃▄▅▆▇█) or as a full plot if the `plotext` library is available.

---

## 16. Limitations & Disclaimers

**This software is provided for educational and research purposes only. It is not financial advice. Trading cryptocurrency perpetual futures involves substantial risk of loss, including the potential loss of all invested capital. Past backtest performance does not guarantee future results.**

### Known Limitations

- **Single asset, single direction**: The system is hardcoded for short-only trading on one symbol. Market regimes that are persistently bullish will result in the system finding no valid entries.
- **No stop-loss**: Open positions are held until the take-profit is reached. In highly adverse conditions, unrealized losses can accumulate without a hard exit mechanism.
- **Backtest overfitting risk**: Despite walk-forward validation and TPE optimization, 1,000 trials across 15 parameters on 14 days of history creates meaningful risk of in-sample overfitting. The 8-hour re-optimization cycle helps adapt to shifting regimes but does not eliminate this risk.
- **Bybit dependency**: The system is tightly coupled to Bybit's V5 REST API. API changes, maintenance windows, or account restrictions will interrupt operation.
- **Fork-based multiprocessing**: Parallel optimization uses `fork` context, which is not compatible with macOS and some Linux configurations when certain libraries (e.g., OpenBLAS) have threading conflicts. Use `spawn` context if issues arise.
- **ML layer is observational**: The `RegimeDetector` and `AnomalyDetector` currently log their outputs but do not gate live trading decisions. Enabling this integration would require additional validation.
- **Small capital design**: The default start balance of $30 USDT means the system is designed for very small accounts. Scaling position sizes without adjusting risk parameters is not recommended without independent analysis.

### Recommended Usage

- Run in paper trading mode first by setting `POSITION_SIZE = 0.0` or using a Bybit testnet API key.
- Monitor the terminal output for at least 2-3 full optimization cycles before running with live funds.
- Do not run on capital you cannot afford to lose entirely.

---

*BACK2BASICS — Back to Basics Short Trading System*
*Repository: [PahtrikProper/BACK-TO-BASICS-SHORT-TRADING](https://github.com/PahtrikProper/BACK-TO-BASICS-SHORT-TRADING)*
