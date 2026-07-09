#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRISHNA OMEGA ULTRA — Backtest completo con datos históricos de OKX
- Descarga automática de velas de 5 minutos (30 días)
- Ejecuta el motor de backtesting
- Genera tablas comparativas con métricas y estadísticas
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ============================================================
# CONFIGURACIÓN
# ============================================================
SYMBOLS = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'AVAX']
DAYS = 30
BAR = '5m'
LIMIT = 100
DATA_DIR = "data/candles_5m"
RESULTS_DIR = "backtest_results"
CAPITAL_INICIAL = 1000.0
TRADE_NOTIONAL = 100.0

# ============================================================
# 1. DESCARGA DE DATOS (API PÚBLICA OKX)
# ============================================================
def get_okx_candles(symbol, after=None):
    inst_id = f"{symbol}-USDT-SWAP"
    url = "https://www.okx.com/api/v5/market/history-candles"
    params = {'instId': inst_id, 'bar': BAR, 'limit': LIMIT}
    if after:
        params['after'] = after
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get('code') != '0':
            return None
        raw = data.get('data', [])
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=['ts','o','h','l','c','vol','vc','vq','cf'])
        df['ts'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms')
        for col in ['o','h','l','c','vol']:
            df[col] = df[col].astype(float)
        return df[['ts','o','h','l','c','vol']]
    except Exception as e:
        return None

def fetch_historical(symbol, days=DAYS):
    start = datetime.now() - timedelta(days=days)
    frames = []
    after = None
    for _ in range(50):
        df = get_okx_candles(symbol, after)
        if df is None or df.empty:
            break
        frames.append(df)
        last_ts = df.iloc[-1]['ts']
        if last_ts < pd.Timestamp(start):
            break
        after = int(last_ts.timestamp() * 1000)
        time.sleep(0.2)
    if not frames:
        return None
    full = pd.concat(frames, ignore_index=True).drop_duplicates('ts').sort_values('ts')
    full = full[full['ts'] >= pd.Timestamp(start)]
    return full

def download_all_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("📥 Descargando datos históricos de OKX...")
    for sym in SYMBOLS:
        df = fetch_historical(sym)
        if df is not None and not df.empty:
            df.to_csv(f"{DATA_DIR}/{sym}.csv", index=False)
            print(f"  ✅ {sym}: {len(df)} velas")
        else:
            print(f"  ⚠️ {sym}: sin datos")
    print(f"Datos guardados en {DATA_DIR}/\n")

# ============================================================
# 2. INDICADORES (VERSIÓN LOCAL PARA BACKTEST)
# ============================================================
def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(df, period):
    tr = pd.concat([
        df['h'] - df['l'],
        abs(df['h'] - df['c'].shift()),
        abs(df['l'] - df['c'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_adx(df, period):
    atr = compute_atr(df, period)
    up = df['h'].diff()
    down = -df['l'].diff()
    plus_dm = up.where((up > down) & (up > 0), 0).rolling(period).mean()
    minus_dm = down.where((down > up) & (down > 0), 0).rolling(period).mean()
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(period).mean()

def compute_ker(close, period):
    abs_diff = abs(close.diff(period))
    sum_abs = close.diff().abs().rolling(period).sum()
    return abs_diff / (sum_abs + 1e-9)

def compute_vwap_zscore(df, period):
    vwap = (df['c'] * df['vol']).rolling(period).sum() / (df['vol'].rolling(period).sum() + 1e-9)
    std = df['c'].rolling(period).std()
    return (df['c'] - vwap) / (std + 1e-9)

def compute_pidelta_score(df, params):
    if len(df) < 50:
        return 0.0
    ker = compute_ker(df['c'], params.get('KER_PERIOD', 10)).iloc[-1]
    vwap_z = compute_vwap_zscore(df, params.get('VWAP_PERIOD', 20)).iloc[-1]
    atr = compute_atr(df, params.get('ATR_PERIOD', 14)).iloc[-1]
    ema_fast = compute_ema(df['c'], params.get('EMA_FAST', 20)).iloc[-1]
    ema_slow = compute_ema(df['c'], params.get('EMA_SLOW', 50)).iloc[-1]
    slope = (ema_fast - ema_slow) / (atr + 1e-9)
    adx = compute_adx(df, params.get('ADX_PERIOD', 14)).iloc[-1]
    mom = df['c'].pct_change(params.get('MOMENTUM_PERIOD', 5)).iloc[-1] * 100
    macro = atr / df['c'].rolling(params.get('MACRO_LOOKBACK', 20)).mean().iloc[-1]
    weights = params.get('PIDELTA_WEIGHTS', {'trend':0.30, 'regime':0.25, 'macro':0.20, 'strength':0.15, 'momentum':0.10})
    raw = (weights['trend'] * np.tanh(slope) +
           weights['regime'] * min(1.0, ker) +
           weights['macro'] * min(1.0, macro) +
           weights['strength'] * min(1.0, adx/40.0) +
           weights['momentum'] * min(1.0, abs(mom)/5.0))
    return np.tanh(raw)

# ============================================================
# 3. SIMULADOR DE BACKTEST
# ============================================================
def run_backtest(data, params, capital=CAPITAL_INICIAL, trade_notional=TRADE_NOTIONAL):
    trades = []
    equity = capital
    cooldown_velas = params.get('COOLDOWN_SECONDS', 900) // 300
    cooldown = {}
    for symbol, df in data.items():
        if df is None or df.empty:
            continue
        for i in range(60, len(df)-1):
            current = df.iloc[:i+1]
            score = compute_pidelta_score(current, params)
            min_score = params.get('MIN_SCORE', 0.40)
            if abs(score) < min_score:
                continue
            adx = compute_adx(current, params.get('ADX_PERIOD', 14)).iloc[-1]
            if adx < params.get('ADX_THRESHOLD', 22):
                continue
            ker = compute_ker(current['c'], params.get('KER_PERIOD', 10)).iloc[-1]
            if ker < params.get('KER_THRESHOLD', 0.50):
                continue
            if symbol in cooldown:
                if (i - cooldown[symbol]) < cooldown_velas:
                    continue
            direction = 'Long' if score > 0 else 'Short'
            entry = df.iloc[i+1]['c']
            atr_val = compute_atr(current, params.get('ATR_PERIOD', 14)).iloc[-1]
            tp_mult = params.get('TP_MULT', 1.8)
            sl_mult = params.get('SL_MULT', 0.9)
            if direction == 'Long':
                tp = entry + atr_val * tp_mult
                sl = entry - atr_val * sl_mult
            else:
                tp = entry - atr_val * tp_mult
                sl = entry + atr_val * sl_mult
            # Simular cierre
            if i+1 < len(df):
                close = df.iloc[i+1]['c']
            else:
                close = df.iloc[-1]['c']
            if direction == 'Long':
                if close >= tp:
                    exit_p = tp
                    result = 'TP'
                elif close <= sl:
                    exit_p = sl
                    result = 'SL'
                else:
                    exit_p = close
                    result = 'Close'
            else:
                if close <= tp:
                    exit_p = tp
                    result = 'TP'
                elif close >= sl:
                    exit_p = sl
                    result = 'SL'
                else:
                    exit_p = close
                    result = 'Close'
            size = trade_notional / entry
            pnl = (exit_p - entry) * size if direction == 'Long' else (entry - exit_p) * size
            pnl -= pnl * 0.0008  # comisiones 0.04% entrada+salida
            equity += pnl
            trades.append({
                'symbol': symbol,
                'direction': direction,
                'entry': entry,
                'exit': exit_p,
                'pnl': pnl,
                'result': result,
                'timestamp': df.iloc[i]['ts']
            })
            cooldown[symbol] = i
    return trades, equity

# ============================================================
# 4. MÉTRICAS Y ESTADÍSTICAS
# ============================================================
def calculate_metrics(trades, initial_capital, final_equity):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    pnls = df['pnl'].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    total_trades = len(pnls)
    win_rate = len(wins)/total_trades*100
    total_win = wins.sum() if len(wins) > 0 else 0
    total_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    expectancy = np.mean(pnls)
    # Drawdown
    eq = np.array([initial_capital] + list(np.cumsum(pnls) + initial_capital))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = dd.max()
    # Sharpe
    ret_daily = pnls / initial_capital
    if len(ret_daily) > 1:
        sharpe = np.mean(ret_daily) / np.std(ret_daily) * np.sqrt(30*24*12)  # anualizado
    else:
        sharpe = 0
    # Sortino (solo riesgo a la baja)
    downside = ret_daily[ret_daily < 0]
    if len(downside) > 0:
        sortino = np.mean(ret_daily) / np.std(downside) * np.sqrt(30*24*12)
    else:
        sortino = float('inf')
    # Calmar
    cagr = ((final_equity / initial_capital) ** (365/30) - 1) * 100 if max_dd > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else float('inf')
    # Duración media (si hay timestamp)
    if 'timestamp' in df.columns:
        durations = df['timestamp'].diff().dt.total_seconds().dropna()
        avg_duration = durations.mean() / 60 if len(durations) > 0 else 0
    else:
        avg_duration = 0
    return {
        'Total Trades': total_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Profit Factor': round(profit_factor, 3),
        'Total PnL (USDT)': round(final_equity - initial_capital, 2),
        'Final Equity (USDT)': round(final_equity, 2),
        'Expectancy (USDT)': round(expectancy, 3),
        'Avg Win (USDT)': round(avg_win, 2),
        'Avg Loss (USDT)': round(avg_loss, 2),
        'Max Drawdown (%)': round(max_dd, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Sortino Ratio': round(sortino, 3),
        'Calmar Ratio': round(calmar, 2),
        'Avg Duration (min)': round(avg_duration, 1),
        'Wins': len(wins),
        'Losses': len(losses),
    }

# ============================================================
# 5. GENERAR TABLAS COMPARATIVAS
# ============================================================
def generate_tables(results):
    print("\n" + "="*70)
    print("📊 TABLA 1 — RENDIMIENTO GLOBAL")
    print("="*70)
    for k, v in results.items():
        if k not in ['trades']:
            print(f"{k:25}: {v}")

    # Desglose por símbolo
    if 'trades' in results:
        df = pd.DataFrame(results['trades'])
        if not df.empty:
            print("\n" + "="*70)
            print("📊 TABLA 2 — DESGLOSE POR ACTIVO")
            print("="*70)
            print(f"{'Activo':<10} {'Trades':>8} {'Win Rate':>12} {'PnL (USDT)':>15}")
            print("-"*50)
            for sym in df['symbol'].unique():
                sub = df[df['symbol'] == sym]
                wr = (sub['pnl'] > 0).sum() / len(sub) * 100
                pnl = sub['pnl'].sum()
                print(f"{sym:<10} {len(sub):>8} {wr:>11.2f}% {pnl:>15.2f}")

            # Resultados por dirección
            print("\n" + "="*70)
            print("📊 TABLA 3 — POR DIRECCIÓN")
            print("="*70)
            for dir_ in ['Long', 'Short']:
                sub = df[df['direction'] == dir_]
                if not sub.empty:
                    wr = (sub['pnl'] > 0).sum() / len(sub) * 100
                    pnl = sub['pnl'].sum()
                    print(f"{dir_:<10} Trades:{len(sub):>4} WinRate:{wr:>6.2f}% PnL:{pnl:>8.2f}")

            # PnL por hora
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                hourly = df.groupby('hour')['pnl'].sum()
                print("\n" + "="*70)
                print("📊 TABLA 4 — PnL POR HORA (UTC)")
                print("="*70)
                for h, pnl in hourly.items():
                    print(f"Hora {h:02d}:00-{h+1:02d}:00  PnL: {pnl:>8.2f} USDT")

# ============================================================
# 6. FUNCIÓN PRINCIPAL
# ============================================================
def main():
    print("🚀 KRISHNA OMEGA ULTRA — BACKTEST COMPLETO")
    # Descargar datos
    download_all_data()
    # Cargar datos
    data = {}
    for sym in SYMBOLS:
        file = f"{DATA_DIR}/{sym}.csv"
        if os.path.exists(file):
            data[sym] = pd.read_csv(file, parse_dates=['ts'])
            print(f"  Cargado {sym}: {len(data[sym])} velas")
        else:
            print(f"  ⚠️ {file} no encontrado")
    if not data:
        print("❌ No hay datos. Saliendo.")
        return
    # Parámetros base (los de config.py)
    params = {
        'TP_MULT': 1.8,
        'SL_MULT': 0.9,
        'MIN_SCORE': 0.40,
        'ADX_THRESHOLD': 22,
        'KER_THRESHOLD': 0.50,
        'ATR_PERIOD': 14,
        'EMA_FAST': 20,
        'EMA_SLOW': 50,
        'KER_PERIOD': 10,
        'VWAP_PERIOD': 20,
        'MOMENTUM_PERIOD': 5,
        'MACRO_LOOKBACK': 20,
        'COOLDOWN_SECONDS': 900,
        'ADX_PERIOD': 14,
        'PIDELTA_WEIGHTS': {'trend':0.30, 'regime':0.25, 'macro':0.20, 'strength':0.15, 'momentum':0.10}
    }
    print("\n▶️ Ejecutando backtest...")
    trades, final_equity = run_backtest(data, params)
    metrics = calculate_metrics(trades, CAPITAL_INICIAL, final_equity)
    metrics['trades'] = trades
    generate_tables(metrics)
    # Guardar resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(trades).to_csv(f"{RESULTS_DIR}/trades.csv", index=False)
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n📁 Resultados guardados en {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
