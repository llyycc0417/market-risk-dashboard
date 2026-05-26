"""S&P500 일별 변동성 예측 — 최종 채택 모델 (HAR-RV-EPU+GPR+Inter).

웹 프론트엔드에서 사용하는 메인 인터페이스:

    from predictor import predict

    result = predict()                       # 캐시 데이터로 빠르게
    result = predict(refresh_data=True)      # yfinance + EPU 새로 다운로드

    result 는 dict (JSON 직렬화 가능):
        latest     — 가장 최근 영업일의 σ̂² 예측 (대시보드 카드용)
        history    — Test 전체 (실제 vs 예측) 시계열 (그래프용)
        metrics    — RMSE / MAE / QLIKE (성능 표시용)
        model      — 모델 이름, 학습 기간, 계수 (about 페이지용)

설계 의도:
    * 모델은 black box. 호출 측은 predict() 만 알면 됨
    * 데이터 갱신 / 학습 / 예측이 한 함수에서 끝남
    * 반환은 plain dict (datetime 은 ISO 문자열로) — json.dumps 호환
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 설정값 (학습된 모델의 핵심 파라미터들)
# ============================================================

HERE = Path(__file__).resolve().parent

# 데이터 소스
TICKER = "^GSPC"                              # S&P500
DEFAULT_START = "1996-01-01"
DEFAULT_END_OFFSET_DAYS = 1                   # 오늘까지
GPR_FILE = HERE / "ai_gpr_data_daily.csv"
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"

# 캐시
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "dataset.csv"

# 모델 정의 (HAR-RV + EPU + GPR + EPU×GPR 상호작용)
GPR_COL = "GPR_AI_sc"
EPU_COL = "EPU_sc"
INTER_COL = "EPU_x_GPR"
FEATURE_COLS = ["log_rv_d", "log_rv_w", "log_rv_m", EPU_COL, GPR_COL, INTER_COL]
TARGET_COL = "target_log_rv"

# Train/Val/Test 분할
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

ANNUALIZATION = 252                            # 영업일 기준


# ============================================================
# 데이터 로드
# ============================================================

def _gkyz_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Garman-Klass-Yang-Zhang single-day realized variance (% 단위)."""
    o, h, l, c = (ohlc[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    c_prev = c.shift(1)
    overnight = np.log(o / c_prev) ** 2
    intraday_hl = 0.5 * (np.log(h / l) ** 2)
    intraday_co = (2.0 * np.log(2.0) - 1.0) * (np.log(c / o) ** 2)
    sigma2 = (overnight + intraday_hl - intraday_co) * (100.0 ** 2)
    sigma2.name = "rv_gkyz"
    return sigma2


def _download_spx(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(TICKER, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    ohlc = raw[["Open", "High", "Low", "Close"]].copy()
    ohlc.index = pd.to_datetime(ohlc.index)

    ret = (np.log(ohlc["Close"] / ohlc["Close"].shift(1)) * 100).rename("return")
    rv = _gkyz_variance(ohlc)
    return pd.concat([ohlc, ret, rv], axis=1)


def _load_gpr() -> pd.DataFrame:
    gpr = pd.read_csv(GPR_FILE, parse_dates=["Date"]).set_index("Date")
    return gpr[["GPR_AI"]].copy()         # 우리 모델은 GPR_AI 만 사용


def _download_epu() -> pd.DataFrame:
    """Baker-Bloom-Davis daily EPU. 큰 CSV 라 재시도 포함."""
    import io
    import urllib.request
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                EPU_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = [c.strip().lower() for c in df.columns]
            df["Date"] = pd.to_datetime(
                df[["year", "month", "day"]]
            )
            return df.set_index("Date")[["daily_policy_index"]].rename(
                columns={"daily_policy_index": "EPU"})
        except Exception as e:
            last_err = e
    raise RuntimeError(f"EPU 다운로드 실패 (3회 시도): {last_err}")


def build_dataset(start: str = DEFAULT_START,
                   end: str | None = None,
                   verbose: bool = False) -> pd.DataFrame:
    """전체 데이터셋 (S&P500 OHLC + return + RV + GPR + EPU + 정규화) 생성.

    누설 차단 처리:
        - GPR/EPU 는 모두 shift(1) 적용 (t-1 정보만 사용)
        - z-score 정규화는 학습셋 (앞 70%) 기준
    """
    if end is None:
        end = (pd.Timestamp.today() + pd.Timedelta(days=DEFAULT_END_OFFSET_DAYS)
               ).strftime("%Y-%m-%d")
    if verbose:
        print(f"  [data] S&P500 다운로드: {start} ~ {end}")
    spx = _download_spx(start, end)

    if verbose:
        print(f"  [data] GPR 로드")
    gpr = _load_gpr()

    if verbose:
        print(f"  [data] EPU 다운로드")
    epu = _download_epu()

    # 합치기
    df = spx.join(gpr, how="left")
    df["GPR_AI"] = df["GPR_AI"].ffill()                # 휴장일 보간
    epu_aligned = epu.sort_index().ffill().reindex(df.index, method="ffill")
    df["EPU"] = epu_aligned["EPU"]

    # t-1 시차 (look-ahead 차단)
    df["GPR_AI"] = df["GPR_AI"].shift(1)
    df["EPU"] = df["EPU"].shift(1)

    df = df.dropna(subset=["return", "rv_gkyz", "GPR_AI", "EPU"])

    # 정규화 (학습셋 기준)
    n_train = int(len(df) * TRAIN_RATIO)

    log_gpr = np.log(df["GPR_AI"].clip(lower=1.0))
    mu_g, sd_g = log_gpr.iloc[:n_train].mean(), log_gpr.iloc[:n_train].std()
    df["GPR_AI_sc"] = (log_gpr - mu_g) / sd_g

    log_epu = np.log(df["EPU"].clip(lower=1.0))
    mu_e, sd_e = log_epu.iloc[:n_train].mean(), log_epu.iloc[:n_train].std()
    df["EPU_sc"] = (log_epu - mu_e) / sd_e

    return df


def load_dataset(refresh_data: bool = False,
                  verbose: bool = False) -> pd.DataFrame:
    """캐시 우선 + refresh_data=True 면 강제 갱신."""
    if CACHE_FILE.exists() and not refresh_data:
        if verbose:
            print(f"  [data] 캐시 로드: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)

    df = build_dataset(verbose=verbose)
    df.to_csv(CACHE_FILE)
    return df


# ============================================================
# HAR-RV 특성 + OLS 학습
# ============================================================

def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    """HAR-RV (daily/weekly/monthly) + EPU + GPR + 상호작용 + 타깃."""
    out = pd.DataFrame(index=df.index)
    log_rv = np.log(df["rv_gkyz"].clip(lower=1e-12))

    out["log_rv_d"] = log_rv.shift(1)
    out["log_rv_w"] = log_rv.shift(1).rolling(5).mean()
    out["log_rv_m"] = log_rv.shift(1).rolling(22).mean()

    out[EPU_COL] = df[EPU_COL]
    out[GPR_COL] = df[GPR_COL]
    out[INTER_COL] = df[EPU_COL] * df[GPR_COL]

    out[TARGET_COL] = log_rv               # 타깃: 같은 시점 t 의 log-RV
    return out.dropna()


def _split(df: pd.DataFrame):
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return (df.iloc[:n_train],
            df.iloc[n_train:n_train + n_val],
            df.iloc[n_train + n_val:])


@dataclass
class FittedModel:
    coefs: dict[str, float]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    feature_cols: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """X 는 self.feature_cols 컬럼을 가진 DataFrame. σ² 단위로 반환."""
        beta0 = self.coefs["const"]
        betas = np.array([self.coefs[c] for c in self.feature_cols])
        pred_log = beta0 + X[self.feature_cols].values @ betas
        return np.exp(pred_log)


def fit_model(df: pd.DataFrame) -> FittedModel:
    """학습셋에서 OLS 적합."""
    feat = _make_features(df)
    train, _, _ = _split(feat)

    X_tr = np.column_stack([np.ones(len(train)), train[FEATURE_COLS].values])
    coef, *_ = np.linalg.lstsq(X_tr, train[TARGET_COL].values, rcond=None)

    return FittedModel(
        coefs=dict(zip(["const"] + FEATURE_COLS, coef.tolist())),
        train_start=train.index.min(),
        train_end=train.index.max(),
        feature_cols=list(FEATURE_COLS),
    )


# ============================================================
# 메트릭
# ============================================================

EPS = 1e-12


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mask = (~np.isnan(actual)) & (~np.isnan(predicted)) & (predicted > 0)
    a, p = actual[mask], predicted[mask]
    if len(a) == 0:
        return {"rmse": float("nan"), "mae": float("nan"),
                "qlike": float("nan"), "n_days": 0}
    return {
        "rmse": float(np.sqrt(np.mean((a - p) ** 2))),
        "mae": float(np.mean(np.abs(a - p))),
        "qlike": float(np.mean(np.log(p + EPS) + a / (p + EPS))),
        "n_days": int(len(a)),
    }


# ============================================================
# 메인 인터페이스
# ============================================================

def predict(refresh_data: bool = False,
            verbose: bool = False) -> dict[str, Any]:
    """최종 모델로 S&P500 변동성 예측 결과를 반환.

    Parameters
    ----------
    refresh_data : bool
        True 면 yfinance + EPU 를 새로 다운로드해 캐시 갱신.
        False 면 캐시 (cache/dataset.csv) 사용. 캐시 없으면 자동 다운로드.
    verbose : bool
        진행 상황 print.

    Returns
    -------
    dict
        {
          "latest": {date, predicted_sigma2, predicted_sigma,
                     predicted_annualized_vol_pct, inputs},
          "history": [{date, actual, predicted}, ...],   # Test 구간
          "metrics": {rmse, mae, qlike, n_days, test_period},
          "model": {name, trained_on, coefficients},
          "generated_at": ISO-8601 timestamp,
        }
    """
    df = load_dataset(refresh_data=refresh_data, verbose=verbose)

    if verbose:
        print(f"  [model] OLS 적합")
    model = fit_model(df)

    # Test 구간 예측
    feat = _make_features(df)
    _, _, test = _split(feat)
    test_pred_sigma2 = model.predict(test)
    test_actual_sigma2 = np.exp(test[TARGET_COL].values)

    metrics = _metrics(test_actual_sigma2, test_pred_sigma2)
    metrics["test_period"] = (
        f"{test.index.min().date()} to {test.index.max().date()}"
    )

    # history 배열 (그래프용)
    history = [
        {"date": d.strftime("%Y-%m-%d"),
         "actual": float(a),
         "predicted": float(p)}
        for d, a, p in zip(test.index, test_actual_sigma2, test_pred_sigma2)
    ]

    # 최신 1개 (대시보드 카드용)
    latest_row = test.iloc[-1]
    latest_pred = float(model.predict(test.iloc[[-1]])[0])
    latest_sigma = float(np.sqrt(latest_pred))
    inputs = {c: float(latest_row[c]) for c in FEATURE_COLS}
    latest_block = {
        "date": test.index[-1].strftime("%Y-%m-%d"),
        "predicted_sigma2": latest_pred,
        "predicted_sigma": latest_sigma,
        "predicted_annualized_vol_pct": float(latest_sigma * np.sqrt(ANNUALIZATION)),
        "inputs": inputs,
    }

    return {
        "latest": latest_block,
        "history": history,
        "metrics": metrics,
        "model": {
            "name": "HAR-RV-EPU+GPR+Inter",
            "trained_on": f"{model.train_start.date()} to {model.train_end.date()}",
            "coefficients": {k: float(v) for k, v in model.coefs.items()},
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# CLI 진입점 (선택)
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Volatility predictor CLI")
    parser.add_argument("--refresh", action="store_true",
                        help="새로 데이터 다운로드 (기본은 캐시 사용)")
    parser.add_argument("--out", default=None,
                        help="결과 JSON 저장 경로 (생략시 stdout)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = predict(refresh_data=args.refresh, verbose=args.verbose)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"saved: {args.out}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()