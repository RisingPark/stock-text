#!/usr/bin/env python3
# watch_writer_v6_krx.py — pykrx를 주 데이터 소스로 사용
# KRX에서 직접 실시간에 가까운 데이터를 가져옴
#
# Features
# - pykrx를 통한 정확한 한국 주식 가격 조회
# - 기술적 지표: GC (골든크로스), RSI, VOL (거래량 급증)
# - 10초마다 업데이트
#
# Usage
#   pip install --upgrade pykrx pandas numpy
#   python watch_writer_v6_krx.py --interval 10

import argparse
import sys
import time
import math
import logging
import warnings
import os
import psutil
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import List, Tuple, Optional, Dict

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pandas as pd
    import numpy as np
    from pykrx import stock
except ImportError:
    print("Install deps:\n  pip install --upgrade pykrx pandas numpy psutil", file=sys.stderr)
    sys.exit(1)

# PID 파일 경로
PID_FILE = Path("watch_writer.pid")

# ---------------------------- IO helpers ----------------------------
def read_pairs(path: Path) -> List[Tuple[str, str]]:
    """Read lines like 'CODE, Alias' from my.txt."""
    if not path.exists():
        path.write_text("123320, 코스피200 레버리지\n233160, 코스닥150 레버리지\n005930, 삼성전자\n000660, SK하이닉스\n", encoding="utf-8")
    pairs: List[Tuple[str, str]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [p.strip() for p in ln.split(",", 1)]
        # Remove .KS/.KQ suffix if present
        code = parts[0].split('.')[0] if '.' in parts[0] else parts[0]
        if len(parts) > 1:
            # Extract alias only (remove target/stop info)
            alias_part = parts[1]
            # Remove target/stop price info if present
            alias = alias_part.split(',')[0].strip()
        else:
            alias = code
        pairs.append((code, alias))
    # de-dup by code, keep first alias
    seen = set()
    uniq = []
    for code, alias in pairs:
        if code not in seen:
            uniq.append((code, alias))
            seen.add(code)
    return uniq

# ---------------------------- Technicals ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic RSI calculation (Wilder's smoothing)."""
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_ewm = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    down_ewm = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = up_ewm / (down_ewm.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def golden_cross_signal(close: pd.Series, short_win: int = 5, long_win: int = 20) -> bool:
    if len(close.dropna()) < (long_win + 2):
        return False
    ma_s = close.rolling(short_win).mean()
    ma_l = close.rolling(long_win).mean()
    # Cross today while not crossed yesterday
    try:
        return bool((ma_s.iloc[-2] <= ma_l.iloc[-2]) and (ma_s.iloc[-1] > ma_l.iloc[-1]))
    except:
        return False

def volume_spike_signal(volume: pd.Series, mult: float = 1.5, window: int = 20) -> bool:
    v = volume.dropna()
    if len(v) < window:
        return False
    try:
        avg20 = v.iloc[-window:].mean()
        today = v.iloc[-1]
        return bool(today >= mult * avg20)
    except:
        return False

# ---------------------------- KRX Data Fetch ----------------------------
def get_krx_quote_and_signals(code: str) -> Tuple[str, float, float, float, float, str, str, str]:
    """
    Return (ticker, price, prev_close, change, pct, currency, exchange, signals_text)
    Using pykrx for accurate Korean stock data
    """
    currency = "KRW"
    exchange = "KRX"
    price = prev_close = change = pct = float("nan")
    signals = []
    
    try:
        # Get today's date
        today = date.today()
        
        # Try to get current price (최신 가격)
        try:
            # 현재가 조회 (오늘 날짜 기준)
            df_today = stock.get_market_ohlcv_by_date(
                today.strftime("%Y%m%d"), 
                today.strftime("%Y%m%d"), 
                code
            )
            if not df_today.empty and '종가' in df_today.columns:
                price = float(df_today['종가'].iloc[-1])
        except:
            pass
        
        # Get historical data for prev_close and technical indicators
        # 60일간의 과거 데이터 조회
        start_date = (today - timedelta(days=60)).strftime("%Y%m%d")
        end_date = today.strftime("%Y%m%d")
        
        df_hist = stock.get_market_ohlcv_by_date(start_date, end_date, code)
        
        if df_hist is not None and not df_hist.empty:
            # Remove today's incomplete data if market is open
            current_hour = datetime.now().hour
            if 9 <= current_hour < 16:  # Korean market hours
                # During market hours, use ticker API for real-time price
                try:
                    ticker_df = stock.get_market_ticker_info(today.strftime("%Y%m%d"))
                    if code in ticker_df.index:
                        price = float(ticker_df.loc[code, '종가'])
                except:
                    pass
            
            # Get closing prices for technical analysis
            closes = df_hist['종가'].dropna()
            volumes = df_hist['거래량'].dropna() if '거래량' in df_hist.columns else pd.Series(dtype=float)
            
            # Calculate prev_close (previous trading day)
            if len(closes) >= 2:
                prev_close = float(closes.iloc[-2])
                # If we don't have today's price yet, use the last available
                if math.isnan(price) and len(closes) > 0:
                    price = float(closes.iloc[-1])
            elif len(closes) == 1:
                price = float(closes.iloc[-1])
                prev_close = price
            
            # Calculate change and percentage
            if not math.isnan(price) and not math.isnan(prev_close) and prev_close != 0:
                change = price - prev_close
                pct = (change / prev_close) * 100.0
            
            # Technical indicators
            if len(closes) >= 20:
                # Golden Cross
                try:
                    if golden_cross_signal(closes):
                        signals.append("BUY:GC")
                except:
                    pass
                
                # RSI
                try:
                    rsi_val = rsi(closes, 14)
                    if len(rsi_val) > 0:
                        latest_rsi = rsi_val.iloc[-1]
                        if not np.isnan(latest_rsi) and latest_rsi < 30:
                            signals.append("BUY:RSI")
                except:
                    pass
                
                # Volume spike
                try:
                    if len(volumes) >= 20 and volume_spike_signal(volumes):
                        signals.append("BUY:VOL")
                except:
                    pass
    
    except Exception as e:
        # Return NaN values on error
        pass
    
    return code, price, prev_close, change, pct, currency, exchange, "/".join(signals) if signals else ""

# ---------------------------- Real-time price fetch ----------------------------
def get_realtime_price(code: str) -> Optional[float]:
    """
    장중 실시간 가격 조회 시도
    """
    try:
        today = date.today().strftime("%Y%m%d")
        # 당일 체결 정보 조회
        df = stock.get_market_ticker_and_trade_info(today, today, code)
        if not df.empty and '종가' in df.columns:
            return float(df['종가'].iloc[-1])
    except:
        pass
    return None

# ---------------------------- Table formatting ----------------------------
def format_table(rows: List[Tuple]) -> str:
    # rows: (Ticker, Alias, Price, PrevClose, Change, %, Cur, Exch, Signal)
    def fmt_num(x, nd=2):
        try:
            if not math.isnan(x):
                return f"{x:,.{nd}f}"
            return "NA"
        except:
            return str(x)
    
    def fmt_change(x):
        try:
            if not math.isnan(x):
                return f"{x:,.0f}"  # 소수점 없이 표시
            return "NA"
        except:
            return str(x)
    
    def fmt_price(x):
        try:
            if not math.isnan(x):
                # 가격이 1000 이상이면 정수로, 미만이면 소수점 2자리
                if x >= 1000:
                    return f"{x:,.0f}"
                else:
                    return f"{x:,.2f}"
            return "NA"
        except:
            return str(x)
    
    header = f"{'Signal':<12} {'Alias':<16} {'Price':>12} {'PrevClose':>12} {'Change':>12} {'%':>8}"
    out = [header, '-' * len(header)]
    for r in rows:
        t, alias, p, pc, ch, pct, cur, ex, sig = r
        # 변화량과 퍼센트에 색상 표시를 위한 기호 추가
        change_str = fmt_change(ch)  # 소수점 없는 포맷 사용
        pct_str = fmt_num(pct)
        if not math.isnan(ch):
            if ch > 0:
                change_str = "+" + change_str
                pct_str = "+" + pct_str
        
        out.append(f"{(sig or ''):<12} {alias:<16} {fmt_price(p):>12} {fmt_price(pc):>12} {change_str:>12} {pct_str:>8}")
    return "\n".join(out)

# ---------------------------- Process Management ----------------------------
def write_pid():
    """현재 프로세스 PID를 파일에 저장"""
    PID_FILE.write_text(str(os.getpid()))

def cleanup_pid():
    """PID 파일 삭제"""
    if PID_FILE.exists():
        PID_FILE.unlink()

def check_running():
    """다른 인스턴스가 실행 중인지 확인"""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text())
            # 프로세스가 실제로 실행 중인지 확인
            if psutil.pid_exists(old_pid):
                try:
                    proc = psutil.Process(old_pid)
                    if 'python' in proc.name().lower():
                        return True
                except:
                    pass
        except:
            pass
        # 오래된 PID 파일 삭제
        PID_FILE.unlink()
    return False

def get_status_indicator():
    """동작 상태 표시 문자 반환"""
    return "● RUNNING" if PID_FILE.exists() else "○ STOPPED"

# ---------------------------- Main loop ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="my.txt")
    ap.add_argument("--out", dest="outfile", default="watch.txt")
    ap.add_argument("--interval", type=int, default=10)
    args = ap.parse_args()
    
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    interval = max(5, args.interval)
    
    # 이미 실행 중인지 확인
    if check_running():
        print("Another instance is already running!")
        print("Check watch.txt for current data.")
        sys.exit(1)
    
    # PID 저장
    write_pid()
    
    print(f"Starting KRX price monitor...")
    print(f"Reading from: {infile}")
    print(f"Writing to: {outfile}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop.\n")
    
    while True:
        try:
            pairs = read_pairs(infile)  # [(code, alias)]
            rows_fmt = []
            
            for code, alias in pairs:
                tk, price, prev_close, change, pct, cur, ex, sig = get_krx_quote_and_signals(code)
                
                # 장중에는 실시간 가격 시도
                current_hour = datetime.now().hour
                if 9 <= current_hour < 16:
                    realtime = get_realtime_price(code)
                    if realtime and not math.isnan(realtime):
                        price = realtime
                        # Recalculate change and pct with new price
                        if not math.isnan(prev_close) and prev_close != 0:
                            change = price - prev_close
                            pct = (change / prev_close) * 100.0
                
                rows_fmt.append((tk, alias, price, prev_close, change, pct, cur, ex, sig))
            
            now = datetime.now(timezone.utc).astimezone()
            status = get_status_indicator()
            header = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {status} | {len(rows_fmt)} tickers (KRX Direct)"
            table = format_table(rows_fmt)
            outfile.write_text(header + "\n" + table + "\n", encoding="utf-8")
            
            # Print to console as well for monitoring
            print(f"\n{header}")
            print(table)
            
        except KeyboardInterrupt:
            print("\nStopping...", file=sys.stderr)
            cleanup_pid()
            print("Stopped.", file=sys.stderr)
            break
        except Exception as e:
            error_msg = f"Error: {e}\n"
            status = get_status_indicator()
            outfile.write_text(f"{status} | {error_msg}", encoding="utf-8")
            print(error_msg, file=sys.stderr)
        
        time.sleep(interval)

if __name__ == "__main__":
    try:
        main()
    finally:
        # 프로그램 종료 시 PID 파일 정리
        cleanup_pid()