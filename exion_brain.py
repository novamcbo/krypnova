import json
import logging
import os
import uuid
import collections
import importlib
import aiofiles
import asyncio
import numpy as np
import random
import inspect
import time
import traceback
from datetime import datetime
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict
from sqlalchemy import select, func
from krypnova.database.database import AsyncSessionLocal,engine
from sqlalchemy.exc import SQLAlchemyError
from krypnova.ai.analysis_modules import (
    ContextAnalyzer, SentimentModel, VolatilityModel,
    LiquidityModel, PatternRecognizer, AnomalyDetector, convert_orderbook_json_to_df,
    SignalEvaluator, OrderBookModel, SpreadMonitor
)
from krypnova.ai.exion_drl_models import PPOAgent, DQNAgent
from krypnova.ai.exion_rl_trainer import ExionRLTrainer
from krypnova.ai.elite_train_models import EliteTrainer
from krypnova.ai.trading_env import TradingEnv
from krypnova.crypto_data import fetch_crypto_data
from krypnova.exchange_connector import ExchangeConnector, detect_exchange
from pattern_detector import analyze_patterns
from krypnova.market_tools import calculate_volatility, calculate_ewma_volatility
from krypnova.advanced_indicators import apply_indicators, calculate_atr
from krypnova.metrics.analysis import calculate_sharpe_ratio,calculate_win_rate, calculate_max_drawdown, calcula_decision_y_confianza, compile_backtest_metrics
from krypnova.models.prediction_log import PredictionLog
from krypnova.trade_risk_manager import (
    calculate_dynamic_stop_loss,
    calculate_dynamic_take_profit,
    calculate_trailing_stop,
    calculate_position_size,
    assess_account_risk,
    max_risk_per_trade,
    calculate_sentiment,
    risk_reward_ratio
)

from krypnova.simulation.backtesting import backtest, process_symbol
from krypnova.simulation.montecarlo import MonteCarloAsyncSimulator
from krypnova.risk_analyzer import RiskAnalyzer, assess_account_risk
from krypnova.ai.exion_defense import ExionDefenseNetwork
from krypnova.utils.market_utils import get_valid_price
from krypnova.tactics.baseline_strategy import BaselineStrategy
from krypnova.portfolio_manager import PortfolioManager
from krypnova.ai.exion_networking_signal import MarketNetworkingSignal
from krypnova.strategy_planner import StrategyPlanner
from typing import List, Dict, Any, Optional
from krypnova.tactics.shadow.shadow_strategy import ShadowStrategy
from krypnova.tactics.sniper.sniper_mode import SniperMode
from krypnova.tactics.bolivar.bolivar_mode import BolivarMode
from krypnova.tactics.hawk_mode import HawkMode
from krypnova.tactics.eclipse_protocol import EclipseProtocol
from krypnova.tactics.counter_tactic import CounterTacticEngine
from krypnova.tactics.phoenix.phoenix_protocol import PhoenixProtocol
from krypnova.tactics.phantom.phantom_tactics import PhantomTactics
from krypnova.tactics.arbitrage import ArbitrageSystem
from krypnova.tactics.hydra.hydra_controller import launch_hydra, simple_hydra_launcher
from krypnova.tactics.gap_close_entry import run_gap_strategy
from krypnova.tactics.sentinel_launcher import  SentinelController
from krypnova.tactics.offensive.momentum import MomentumDriver
from krypnova.tactics.offensive.trap_reversal import TrapReversalEngine
from krypnova.tactics.offensive.stop_hunt_executor import StopHuntExecutor
from krypnova.tactics.offensive.liquidity_sweeper import LiquiditySweeper
from krypnova.tactics.starks.profiles.rebound_entry import ReboundEntry
from krypnova.tactics.starks.profiles.memecoin_sniper import MemecoinSniper
from krypnova.tactics.starks.profiles.htf_breakout import HTFBreakoutEntry
from krypnova.tactics.starks.profiles.news_reversal import NewsReversalEntry
from krypnova.tactics.evasion_manager import EvasionManager
from krypnova.tactics.scalping.gamma_scalping import GammaScalper
from krypnova.utils.reaction_metrics import compute_decision_confidence,recommend_tp_sl
from krypnova.tactics.pattern_attack_executor import PatternAttackExecutor
from krypnova.models.strategy_performance import StrategyPerformanceLog
from krypnova.ai.portfolio_optimizer import PortfolioOptimizer
from analytics.top_movers import TopMoversFetcher
from analytics.hidden_opportunities import detect_hidden_gems_advanced
from offensive_selector import OffensiveSelector
from pattern_detector import evaluate_patterns_score, analyze_patterns
from combine_roi import combine_signals_roi
from krypnova.tactics.proxys.proxy_manager import ProxyManager
from krypnova.trades.executor import TradeExecutor
from krypnova.portfolio_manager import PortfolioManager
from krypnova.models.portfolio import Portfolio
from krypnova.clones_manager import ClonesManager
from krypnova.models.decision import  DecisionLog
EXECUTABLE_ACTIONS = {
    "buy",
    "sell",
    "place limit order",
    "scalping",
    "short",
    "long",
    "stop loss", "stop_loss",
    "trailing stp", "trailing_stp",
    "take profit", "take_profit",
    "arbitrage",
    "hold",
    "wait"
}

def is_executable_action(action: str) -> bool:
    norm = (action or "").replace("_", " ").strip().lower()
    return norm in EXECUTABLE_ACTIONS

ctx = {
    "profile": "moderate",
    "capital": 10_000,
    "asset": "BTC-USD",           # y actualiza en cada loop si quieres
    "connector": ExchangeConnector(),
    "clones_manager": ClonesManager(),
    "liquidity": 500000,
    "volume_24h": 1000000,
    "volatility": 0.05,
    "user_capital": 10_000,
    "ticker": "BTC-USD",
    "session": "mysession",
    "assets": ["BTC-USD"],
    "email": "user@email.com",
}
def _safe_init(Cls, **kwargs):
    """
    Instancia Cls filtrando kwargs a los parámetros que realmente soporta.
    Si la inspección falla, intenta Cls() sin kwargs.
    """
    try:
        sig = inspect.signature(Cls)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return Cls(**filtered)
    except Exception:
        return Cls()



def init_db():
    Base.metadata.create_all(engine)

logger = logging.getLogger("ExionBrain")
logger.setLevel(logging.INFO)

import numpy as np

def _combine_roi_estimates(*, montecarlo_result: dict | None, montecarlo_raw, backtest_result: dict | None, start_price: float, signals_info: dict | None) -> float:
    """
    Devuelve un ROI total (float) combinando:
      - Monte Carlo: (expected_value / start_price) - 1
      - Backtest: backtest_result['roi'] o ['total_return'] o ['cagr'] si existen
      - Señales: signal_strength como último recurso
    """
    candidates = []

    # 1) Monte Carlo (resumen)
    mc = montecarlo_result or {}
    try:
        ev = mc.get("expected_value", None)
        if ev is not None and start_price > 0:
            candidates.append(float(ev) / float(start_price) - 1.0)
    except Exception:
        pass

    # 1b) Monte Carlo (matriz cruda), por si no hay resumen
    if (not candidates) and (montecarlo_raw is not None):
        try:
            arr = np.asarray(montecarlo_raw)
            if arr.ndim == 2 and arr.shape[1] > 0 and start_price > 0:
                final_prices = arr[:, -1]
                if final_prices.size > 0:
                    candidates.append(float(np.mean(final_prices) / float(start_price) - 1.0))
            elif arr.ndim == 1 and arr.size > 0 and start_price > 0:
                candidates.append(float(np.mean(arr) / float(start_price) - 1.0))
        except Exception:
            pass

    # 2) Backtest
    bt = backtest_result or {}
    for k in ("roi", "total_return", "cagr"):
        try:
            v = bt.get(k, None)
            if isinstance(v, (int, float)):
                candidates.append(float(v))
                break
        except Exception:
            pass

    # 3) Señales (último recurso)
    try:
        s = (signals_info or {}).get("signal_strength", None)
        if isinstance(s, (int, float)):
            candidates.append(float(s))
    except Exception:
        pass

    # Media robusta
    try:
        candidates = [float(x) for x in candidates if x is not None and np.isfinite(x)]
        if not candidates:
            return 0.0
        return float(np.mean(candidates))
    except Exception:
        return 0.0

def summarize_montecarlo(mc_matrix, start_price: float) -> dict:
    try:
        arr = np.asarray(mc_matrix)
        finals = arr[:, -1] if (arr.ndim == 2 and arr.shape[1] > 0) else (arr if arr.ndim == 1 else None)
        if finals is None: return {}
        finals = finals[np.isfinite(finals)]
        if finals.size == 0 or start_price <= 0: return {}
        prob_up = float((finals > start_price).mean())
        return {
            "expected_value": float(finals.mean()),
            "median": float(np.median(finals)),
            "max": float(finals.max()),
            "min": float(finals.min()),
            "std": float(finals.std(ddof=1)) if finals.size > 1 else 0.0,
            "probabilidad_ganancia": prob_up,   # <-- CLAVE PARA EL GATE
            "n_sims": int(finals.size),
            "start_price": float(start_price),
        }
    except Exception:
        return {}
# Evita duplicados si ya hay handlers
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)




STATE_SIZE = 10          # or whatever value matches your environment
ACTION_SIZE = 3

import asyncio


async def _safe_execute_order(executor, *, side: str, symbol: str,
                              quantity: float, price: float | None = None,
                              order_kind: str = "market", **extra):
    """
    Llama executor.execute_order con los nombres de argumentos que realmente soporte:
    quantity/qty/size/amount y type/order_type/ord_type. Soporta métodos async o sync.
    """
    if not hasattr(executor, "execute_order"):
        raise AttributeError("executor no tiene execute_order")

    fn = executor.execute_order
    sig = inspect.signature(fn)
    params = set(sig.parameters.keys())

    payload = {}

    # side
    for k in ("side", "action", "order_side"):
        if k in params:
            payload[k] = side
            break

    # symbol / market / instrument
    for k in ("symbol", "market", "instrument", "pair"):
        if k in params:
            payload[k] = symbol
            break

    # quantity mapeado
    size_field = None
    for k in ("quantity", "qty", "size", "amount", "base_qty", "base_amount"):
        if k in params:
            size_field = k
            break
    if size_field:
        payload[size_field] = quantity
    else:
        # no hay campo de tamaño soportado → lo dejamos sin tamaño (algunos executors lo calculan por %)
        logger.warning("execute_order no expone parámetro de tamaño; se enviará sin 'quantity'.")

    # tipo de orden
    if "type" in params:
        payload["type"] = order_kind
    elif "order_type" in params:
        payload["order_type"] = order_kind
    elif "ord_type" in params:
        payload["ord_type"] = order_kind

    # precio (si aplica)
    if price is not None:
        for k in ("price", "limit_price"):
            if k in params:
                payload[k] = price
                break

    # añade extras que hagan match con la firma
    for k, v in (extra or {}).items():
        if k in params:
            payload[k] = v

    logger.info(f"[AutoExec] execute_order payload normalizado: {payload}")

    try:
        res = fn(**payload)
        if inspect.isawaitable(res):
            res = await res
        return res
    except TypeError as e:
        logger.error(f"Firma incompatible al llamar execute_order: {e} | firma={sig}")
        raise
async def _safe_calculate_metrics(strat, symbol, df, capital):
    try:
        # La mayoría de tus estrategias deben implementar calculate_metrics como async
        if hasattr(strat, "calculate_metrics"):
            return await strat.calculate_metrics(symbol, df, capital)
        # fallback: si no es async
        elif callable(getattr(strat, "calculate_metrics", None)):
            return strat.calculate_metrics(symbol, df, capital)
        else:
            return 0, 0, {}, {}
    except Exception as e:
        return 0, 0, {}, {}

async def _maybe_await(obj):
    if asyncio.iscoroutine(obj):
        return await obj
    return obj
def safe_strategy_instance(cls, context, symbol):
    """
    Instancia una estrategia pasando solo los argumentos requeridos.
    Si faltan argumentos obligatorios, devuelve None y muestra una advertencia.
    """
    # Mapea argumentos requeridos por clase
    required_args_map = {
        "CounterTacticEngine": ["asset"],
        "GapCloserEntry": ["connector"],
        "SentinelController": ["clones_manager"],
        "TacticalPowerEngine": ["liquidity", "volume_24h", "volatility", "user_capital"],
        # ...agrega los que correspondan...
    }
    class_name = cls.__name__
    base_kwargs = {
        "user_id": context.get("user_id"),
        "capital": context.get("capital"),
        "profile": context.get("profile"),
        "symbol": symbol,
        "exchange": context.get("exchange"),
        "exchange_name": context.get("exchange_name"),
        "session": context.get("session"),
    }
    # Añade los requeridos específicos si están en el mapping
    for arg in required_args_map.get(class_name, []):
        base_kwargs[arg] = context.get(arg)
    # Verifica que tienes TODOS los requeridos
    sig = inspect.signature(cls)
    missing = [k for k in sig.parameters if sig.parameters[k].default is inspect.Parameter.empty and k not in base_kwargs]
    if missing:
        print(f"[{class_name}] No se puede instanciar: faltan {missing}")
        return None
    try:
        return cls(**base_kwargs)
    except Exception as e:
        print(f"Could not instantiate {cls}: {e}")
        return None


def _build_strategy_instance(Strat, context, capital, symbol):
    """
    Instancia una estrategia con los argumentos más comunes que pueda aceptar.
    """
    common_kwargs = {
        "user_id": context.get("user_id", "autotest") if context else "autotest",
        "capital": capital,
        "profile": context.get("profile", "moderate") if context else "moderate",
        "symbol": symbol,
        "exchange": context.get("exchange", "kraken") if context else "kraken",
        "exchange_name": context.get("exchange", "kraken") if context else "kraken",
        "session": context.get("session", None) if context else None
    }
    try:
        sig = inspect.signature(Strat)
        args_to_pass = {k: v for k, v in common_kwargs.items() if k in sig.parameters}
        return Strat(**args_to_pass)
    except Exception as e:
        try:
            return Strat()  # fallback sin argumentos
        except Exception:
            return None


# === Hashability & ROI sanitizers ===
def _freeze_hashable(x):
    """
    Convierte estructuras a hashables para sets/dicts:
    - list/tuple -> tuple (recursivo)
    - dict -> tuple ordenada (clave_str, valor_freeze)
    - set -> tuple ordenada
    """
    if isinstance(x, dict):
        return tuple(sorted((str(k), _freeze_hashable(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_freeze_hashable(v) for v in x)
    if isinstance(x, set):
        return tuple(sorted(_freeze_hashable(v) for v in x))
    return x

def sanitize_name(name):
    """
    Normaliza el nombre a string:
      - Lista/tupla: toma el primer elemento no vacío (recursivo).
      - None -> "".
      - Otros: str(name).
    """
    while isinstance(name, (list, tuple)):
        if not name:
            return ""
        name = name[0]
    if name is None:
        return ""
    return str(name)

def _sanitize_signal(sig: dict) -> dict:
    """
    Señal limpia para combinación ROI (solo tipos primitivos):
      - 'name', 'strategy', 'symbol' -> str
      - 'signals_used' -> list[str]
      - 'roi', 'confidence' -> float o None
    """
    s = dict(sig or {})
    for k in ("name", "strategy", "symbol"):
        if k in s and not isinstance(s[k], (str, int, float)):
            s[k] = sanitize_name(s[k])
        elif k in s:
            s[k] = str(s[k])
    if "signals_used" in s:
        su = s["signals_used"]
        if isinstance(su, (set, tuple)):
            su = list(su)
        elif not isinstance(su, list):
            su = [su]
        s["signals_used"] = [sanitize_name(x) for x in su]
    for k in ("roi", "confidence"):
        if k in s and s[k] is not None:
            try:
                s[k] = float(s[k])
            except Exception:
                s[k] = None
    return s
def make_strategies(context, symbol):
    import inspect
    from offensive_selector import OffensiveSelector
    selector = OffensiveSelector()
    STRATEGY_REQUIRED_ARGS = {
        "counter_tactic": ["asset", "user_id"],
        "gap_close_entry": ["connector"],
        "sentinel_launcher": ["clones_manager", "user_id"],
        "tactical_power_engine": ["liquidity", "volume_24h", "volatility", "user_capital"],
        "arbitrage": ["user_id", "ticker"],
        "eclipse_protocol": ["user_id", "session"],
        "memecoin_sniper": ["user_id", "assets"],
        "phoenix_protocol": ["user_id", "email"]
    }
    def get_strategy_args(strat_name, context, symbol):
        reqs = STRATEGY_REQUIRED_ARGS.get(strat_name, [])
        args = {}
        missing = []
        for k in reqs:
            if k == "user_id":
                args[k] = context.get("user_id")
            elif k == "asset":
                args[k] = context.get("asset") or symbol
            elif k == "connector":
                args[k] = context.get("connector")
            elif k == "clones_manager":
                args[k] = context.get("clones_manager")
            elif k == "liquidity":
                args[k] = context.get("liquidity")
            elif k == "volume_24h":
                args[k] = context.get("volume_24h")
            elif k == "volatility":
                args[k] = context.get("volatility")
            elif k == "user_capital":
                args[k] = context.get("user_capital")
            elif k == "ticker":
                args[k] = context.get("ticker") or symbol
            elif k == "session":
                args[k] = context.get("session")
            elif k == "assets":
                args[k] = context.get("assets") or [symbol]
            elif k == "email":
                args[k] = context.get("email")
            else:
                args[k] = context.get(k)
            if args[k] is None:
                missing.append(k)
        return args, missing

    strategies = []
    for strat_name, Strat in selector.available_strategies.items():
        common_kwargs = {
            "user_id": context.get("user_id"),
            "capital": context.get("capital"),
            "profile": context.get("profile"),
            "symbol": symbol,
            "exchange": context.get("exchange"),
            "exchange_name": context.get("exchange_name"),
            "session": context.get("session")
        }
        args, missing = get_strategy_args(strat_name, context, symbol)
        if missing:
            print(f"[{strat_name}] No se puede instanciar: faltan {missing}")
            continue
        try:
            sig = inspect.signature(Strat)
            args_to_pass = {k: v for k, v in {**common_kwargs, **args}.items() if k in sig.parameters}
            instance = Strat(**args_to_pass)
            if hasattr(instance, "calculate_metrics") and callable(instance.calculate_metrics):
                strategies.append(instance)
        except Exception as e:
            print(f"Could not instantiate {Strat}: {e}")
    return strategies

def universal_instantiator(cls, args_dict):
    """
    Instancia cualquier clase 'cls' usando solo los argumentos que acepta el constructor.
    Si el constructor acepta **kwargs, se pasan todos los argumentos recibidos.
    Si faltan argumentos obligatorios, lanzará el error normal de Python.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # Verificar si acepta **kwargs
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())

    # Argumentos válidos según el signature (descartando self)
    valid_args = {k: v for k, v in args_dict.items() if k in params and k != 'self'}

    if accepts_kwargs:
        # Pasa todos los argumentos si acepta **kwargs
        return cls(**args_dict)
    else:
        # Solo los argumentos explícitos
        return cls(**valid_args)
def _sanitize_capital_dict(capital_dict: dict) -> dict:
    """Claves str y valores float para capital por estrategia."""
    if not isinstance(capital_dict, dict):
        return {}
    out = {}
    for k, v in capital_dict.items():
        try:
            out[sanitize_name(k)] = float(v)
        except Exception:
            out[sanitize_name(k)] = 0.0
    return out
def _sanitize_name(name):
    """Normaliza el nombre para usarlo como clave: minúsculas, sin espacios."""
    return str(name).replace(" ", "_").lower() if name else ""


async def _maybe_await(obj):
    """Espera el objeto si es awaitable, si no lo retorna directo."""
    if asyncio.iscoroutine(obj):
        return await obj
    return obj


async def _safe_calculate_metrics(strat, symbol, df, capital):
    cm = getattr(strat, "calculate_metrics", None)
    if not callable(cm):
        raise AttributeError("calculate_metrics no disponible")
    out = cm(symbol, df, capital)
    out = await _maybe_await(out)
    if (not isinstance(out, (list, tuple))) or len(out) != 4:
        raise TypeError("calculate_metrics debe devolver (roi, mc_roi, metrics, mc_results)")
    return out

async def _safe_action_and_confidence(strat, df, metrics, mc_results):
    gac = getattr(strat, "get_action_and_confidence", None)
    if not callable(gac):
        return "HOLD", 0.0
    res = gac(df, metrics, mc_results)
    res = await _maybe_await(res)
    try:
        action, confidence = res
    except Exception:
        action, confidence = "HOLD", 0.0
    return action, confidence
async def _safe_combine_signals_roi(signals, capital):
    """
    Wrapper robusto para combine_signals_roi:
      - Sanea señales y capital.
      - Fallback a estructuras 100% hashables si algo interno usa sets/dicts.
    """
    from combine_roi import combine_signals_roi
    signals = [_sanitize_signal(s) for s in (signals or [])]
    capital = _sanitize_capital_dict(capital or {})

    # --- PATCH DEBUG: verifica que no hay listas como claves o en fields críticos ---
    for idx, s in enumerate(signals):
        if isinstance(s.get("name"), list):
            raise Exception(f"[FATAL _safe_combine_signals_roi] Signal at idx={idx} has name as LIST: {s.get('name')} - Señal completa: {s}")
    for k in capital.keys():
        if isinstance(k, list):
            raise Exception(f"[FATAL _safe_combine_signals_roi] Capital key is list: {k} - Capital dict: {capital}")

    # También revisa si signals es lista de listas por error
    if any(isinstance(s, list) for s in signals):
        raise Exception(f"[FATAL _safe_combine_signals_roi] signals contiene sublistas: {signals}")

    try:
        return await combine_signals_roi(signals, capital=capital)
    except TypeError as e:
        # Forzamos estructuras hashables (tuplas) como último recurso
        frozen_signals = [_freeze_hashable(s) for s in signals]
        frozen_capital = _freeze_hashable(capital)
        raise Exception(f"[FATAL _safe_combine_signals_roi] TypeError en combine_signals_roi: {e}\n"
                        f"signals={signals}\ncapital={capital}\n"
                        f"frozen_signals={frozen_signals}\nfrozen_capital={frozen_capital}")
        # Si quieres intentar igualmente, comenta el raise y descomenta abajo:
        # return await combine_signals_roi(frozen_signals, capital=frozen_capital)
def _num(x, *keys):
    """Extrae un float de x. Si es dict, intenta por claves; si es lista/tupla, toma el primer numérico."""
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, dict):
            for k in keys:
                if k in x and isinstance(x[k], (int, float, np.number)):
                    return float(x[k])
            # fallback: primer valor numérico en el dict
            for v in x.values():
                if isinstance(v, (int, float, np.number)):
                    return float(v)
            return None
        if isinstance(x, (list, tuple)) and len(x) > 0:
            for v in x:
                if isinstance(v, (int, float, np.number)):
                    return float(v)
            return None
        return float(x)
    except Exception:
        return None

def make_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    else:
        return obj

def dict_guard(var, default=None, varname="variable", logger=None):
    """
    Asegura que 'var' sea un dict. Si es un set u otro tipo, lo reemplaza por 'default' y loguea el error.
    """
    if not isinstance(var, dict):
        if logger:
            logger.error(f"[BUG] {varname} no es dict. Es {type(var)}: {var}")
        print(f"[BUG] {varname} no es dict. Es {type(var)}: {var}")
        return default if default is not None else {}
    return var
def print_df_debug(df, symbol):
    print(f"\n=== DEBUG OHLCV DataFrame para {symbol} ===")
    print("¿Es DataFrame?:", isinstance(df, pd.DataFrame))
    if isinstance(df, pd.DataFrame):
        print("Shape:", df.shape)
        print("Columnas:", df.columns.tolist())
        print("Nulos por columna:", df.isnull().sum().to_dict())
        print("¿Vacío?:", df.empty)
        print("Primeras filas:\n", df.head())
    else:
        print("Valor recibido:", df)
    print("=============================\n")
    from collections import defaultdict
    import logging
async def save_decision_db(self, user_id, symbol, decision, context, result):
    """
    Guarda la decisión en la base de datos usando AsyncSessionLocal y el modelo DecisionLog.
    """
    from krypnova.models.decision import DecisionLog  # Ajusta el import al path real
    try:
        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                log = DecisionLog(
                    user_id=user_id,
                    symbol=symbol,
                    decision=decision,
                    context_json=json.dumps(context, default=str),
                    result_json=json.dumps(result, default=str),
                    timestamp=datetime.utcnow()
                )
                session.add(log)
    except Exception as e:
        self.logger.error(f"[DB] Error guardando DecisionLog: {e}")
class ExionBrain:
    def __init__(self, memory_path="exion_memory.json", user_id="default_user", profile="moderate", capital=100000):
            # --- deps de BD / sesiones ---
            self.AsyncSessionLocal = AsyncSessionLocal  # asegúrate que esté importado
            self.proxy_manager = ProxyManager()


            # --- RL / Entrenadores (requieren STATE_SIZE/ACTION_SIZE definidos) ---
            self.dqn_agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
            self.ppo_agent = PPOAgent(state_dim=STATE_SIZE, action_dim=ACTION_SIZE)
            self.rl_trainer = ExionRLTrainer(strategy_profile=profile)
            self.elite_trainer = EliteTrainer()

            # --- estado / memoria ---
            self.memory_path = memory_path
            self.memory: dict = {}  # para logs/estados persistentes
            self.chat_history: list = []  # historial de chat
            self.decision_history: list = []  # usado por defense_decision

            # --- modelos / detectores ---
            self.context_analyzer = ContextAnalyzer()
            self.sentiment_model = SentimentModel()
            self.volatility_model = VolatilityModel()
            self.liquidity_model = LiquidityModel()
            self.pattern_recognizer = PatternRecognizer()
            self.anomaly_detector = AnomalyDetector()
            self.signal_evaluator = SignalEvaluator()
            self.orderbook_model = OrderBookModel()
            self.spread_monitor = SpreadMonitor()

            # --- conectores / sistemas ---
            self.exchange = ExchangeConnector()
            self.portfolio = _safe_init(PortfolioManager, user_id=user_id, capital=capital)
            self.executor = TradeExecutor(portfolio=self.portfolio)
            self.defense = ExionDefenseNetwork()
            self.pattern_executor = PatternAttackExecutor(capital=capital)
            self.arbitrage_system = ArbitrageSystem()  # ← ahora sí comentado si quieres

            # --- config de usuario / sesión ---
            self.user_id = user_id
            self.profile = profile
            self.capital = capital
            self.symbol_memory = defaultdict(lambda: {"roi": 0.0, "decision": "hold", "risk": 0})
            self.default_lang = "en"


            self.networking_signal = _safe_init(MarketNetworkingSignal, profile=profile)
            self.strategy_planner = _safe_init(StrategyPlanner, user_id=user_id, capital=capital, profile=profile)
            self.offensive_selector = OffensiveSelector(user_profile=self.profile, user_id=self.user_id)

            # --- logger ---
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                self.logger.addHandler(h)

            # --- hook opcional usado por defense_decision ---
            if not hasattr(self, "autoajuste_defensa"):
                self.autoajuste_defensa = lambda: None

    def train_traditional(self, X_train, y_train, X_val, y_val):
        "Trains traditional models (XGB, LSTM, CNN, etc.) and returns metrics."
        results = self.elite_trainer.train(X_train, y_train, X_val, y_val)
        return results

    async def train_rl(self, episodes=50, batch_size=32, patience=10, window=5):
        """
        Trains the RL model with early stopping.
        """
        await self.rl_trainer.train(
            episodes=episodes,
            batch_size=batch_size,
            early_stopping_patience=patience,
            early_stopping_window=window
        )

    async def periodic_retraining(exion, interval_hours=24):
        while True:
            print(f"[{datetime.now()}] 🔄 Reentrenando modelos tradicionales y RL...")
            await exion.elite_trainer.train_models()
            await exion.train_rl(episodes=100)
            await asyncio.sleep(interval_hours * 3600)



    async def get_last_trained_log_id(self):
        try:
            async with aiofiles.open("last_trained_id.txt", "r") as f:
                content = await f.read()
                return int(content)
        except FileNotFoundError:
            return 0

    async def set_last_trained_log_id(self, value):
        async with aiofiles.open("last_trained_id.txt", "w") as f:
            await f.write(str(value))

    async def retrain_on_new_samples(self, min_new_samples=500):
        last_trained_id = await self.get_last_trained_log_id()
        while True:
            new_samples = await self.count_new_logs_since(last_trained_id)
            if new_samples >= min_new_samples:
                print(f"🔁 Hay {new_samples} nuevas muestras, reentrenando...")
                await self.elite_trainer.train_models()
                await self.train_rl(episodes=100)
                last_trained_id = await self.get_max_log_id()
                await self.set_last_trained_log_id(last_trained_id)
            await asyncio.sleep(3600)
    async def count_new_logs_since(self, last_id):
        async with self.AsyncSessionLocal() as session:  # O como crees la sesión asíncrona
            result = await session.execute(
                select(func.count()).select_from(StrategyPerformanceLog).where(StrategyPerformanceLog.id > last_id)
            )
            count = result.scalar_one()
            return count

    async def get_max_log_id(self):
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(func.max(StrategyPerformanceLog.id))
            )
            max_id = result.scalar_one() or 0
            return max_id


    async def load_memory(self):
        def _read_memory(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                return {}  # Si el archivo no existe, comienza con memoria vacía
            except json.JSONDecodeError:
                return {}  # Si está corrupto, también empieza vacío

        self.memory = await asyncio.to_thread(_read_memory, self.memory_path)

    async def save_memory(self):
        def _write_memory(path, data):
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(make_serializable(data), f, indent=2)

        await asyncio.to_thread(_write_memory, self.memory_path, self.memory)

    async def record_action(self, user_id, mode, context, decision, result):
        await self.load_memory()

        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
            "context": context,
            "decision": decision,
            "result": result
        }

        if user_id not in self.memory:
            self.memory[user_id] = []

        self.memory[user_id].append(log)
        await self.save_memory()

    async def learn_from_outcome(self, user_id):
        print(f"🧠 (async) Aprendiendo del usuario {user_id}...")

    async def interact(self, input_text: str | None = None, use_voice: bool = False):
        # 1) Entrada
        if use_voice:
            audio_file = record_audio(duration=6)
            result = transcribe_audio(audio_file)
            input_text = (result.get("text") or "").strip()
            lang = result.get("language", "en")
        else:
            lang = "en"  # o detecta automáticamente si quieres

        if not input_text:
            return "No recibí texto para procesar."

        # 2) Historial (usuario)
        self.chat_history.append({"role": "user", "text": input_text})
        # Si quieres mantener memory como duplicado del historial, descomenta:
        # self.memory.append({"role": "user", "text": input_text})

        # 3) Analítica
        sentiment = analyze_sentiment(input_text)  # sync
        risk_level = await evaluate_risk_level(input_text)  # async

        # 4) Respuesta
        response = self.generate_response(input_text, sentiment)

        # 5) Historial (modelo)
        self.chat_history.append({"role": "exion", "text": response})
        # self.memory.append({"role": "exion", "text": response})

        # 6) Traducción / voz / alertas
        translated = translate_response(response, lang)  # si es async: translated = await translate_response(...)
        self.logger.info(f"🤖 Exion: {translated}")

        if use_voice:
            speak(translated, lang)

        if risk_level in ("alto", "crítico"):
            self.send_alert(translated, lang)

        return translated

    def generate_response(self, text: str, sentiment: str) -> str:
        # trabaja SIEMPRE sobre chat_history
        last3 = self.chat_history[-3:]
        last2 = self.chat_history[-2:]

        try:
            if any("quién eres" in (x.get("text", "").lower()) for x in last3):
                return "Soy Exion, tu asistente de análisis inteligente. ¿En qué puedo ayudarte hoy?"

            if "btc" in text.lower() and any("precio" in (x.get("text", "").lower()) for x in last2):
                return "Puedo analizar el precio actual de BTC. ¿Quieres que lo revise en Kraken o Binance?"

            if "gracias" in text.lower():
                return "¡Siempre a la orden! ¿Hay algo más en lo que te pueda asistir?"

            if sentiment == "positive":
                return "¡Me alegra escuchar eso! ¿Quieres analizar algún activo o recibir alguna recomendación?"

            if sentiment == "negative":
                return "Lamento escuchar eso. ¿Quieres que revise el mercado por ti?"

            return "Estoy procesando tu solicitud, dame un momento para ayudarte mejor."

        except Exception as e:
            self.logger.warning(f"[generate_response] Fallback por error: {e}")
            return "Estoy procesando tu solicitud, dame un momento para ayudarte mejor."

    def send_alert(self, message: str, lang: str):
        logger.warning(f"⚠️ Alerta de riesgo: {message}")
        speak(f"⚠️ Atención: {message}", lang)

    def get_memory(self):
        return self.memory

    async def evaluate_detection_risk(self, user_id):
        await self.load_memory()
        recent = [a for a in self.memory.get(user_id, []) if a["mode"] == "phantom"]
        psi = len(recent) * 2
        return min(psi, 100)

    async def evaluate_patterns_score(df: pd.DataFrame) -> float:
        try:
            patterns = await analyze_patterns(df)
            if not patterns:
                return 0.0

            # Clasifica patrones
            bullish = {"Bullish Engulfing", "Hammer", "Triangle", "Cup and Handle", "Channel Up", "Wedge Up"}
            bearish = {"Bearish Engulfing", "Shooting Star", "Head and Shoulders", "Double Top", "Wedge Down",
                       "Channel Down"}

            score = 0
            for p in patterns:
                name = p.get("pattern", "").lower()
                if any(b in name for b in bullish):
                    score += 1.0
                elif any(b in name for b in bearish):
                    score -= 1.0

            return score
        except Exception as e:
            logger.warning(f"[ExionBrain] ⚠️ Error en evaluate_patterns_score: {e}")
            return 0.0

    def sharpe_ratio(returns, risk_free_rate=0.0):
        excess_returns = np.array(returns) - risk_free_rate
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)

    async def optimize_strategy(df: pd.DataFrame, n_trials: int = 50) -> dict:
        df = df.copy().dropna()

        if df.empty or len(df) < 50:
            return {"error": "Not enough data for optimization."}

        def objective(trial, df_local: pd.DataFrame):
            atr_period = trial.suggest_int("atr_period", 7, 21)
            sl_mult = trial.suggest_float("stop_loss_multiplier", 1.0, 3.0)
            tp_mult = trial.suggest_float("take_profit_multiplier", 1.0, 5.0)

            df_local = df_local.copy()
            df_local["returns"] = df_local["close"].pct_change().fillna(0)
            df_local["rolling_atr"] = df_local["high"].rolling(atr_period).max() - df_local["low"].rolling(
                atr_period).min()
            df_local = df_local.dropna(subset=["rolling_atr"])

            if df_local.empty:
                raise optuna.exceptions.TrialPruned()

            df_local["signal"] = np.where(df_local["returns"] > 0.002, 1, 0)
            sl = df_local["close"] - sl_mult * df_local["rolling_atr"]
            tp = df_local["close"] + tp_mult * df_local["rolling_atr"]
            df_local["exit_price"] = np.where(df_local["signal"] == 1, tp, sl)
            df_local["trade_return"] = np.where(df_local["signal"] == 1, df_local["exit_price"] / df_local["close"] - 1,
                                                0)

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_index, _ in tscv.split(df_local):
                segment = df_local.iloc[train_index]
                score = ExionBrain.sharpe_ratio(segment["trade_return"].values)
                if np.isnan(score) or np.isinf(score):
                    raise optuna.exceptions.TrialPruned()
                scores.append(score)

            return np.mean(scores)

        # 👇 Pasa `df` usando lambda
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.warning("⚠️ Ningún trial completado. Usando hiperparámetros por defecto.")
            return {
                "atr_period": 14,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 3.0
            }

        best_params = study.best_params
        best_params["score"] = study.best_value
        return best_params


    def ejecutar_estrategia_baseline(symbol, context=None):
        baseline = BaselineStrategy()
        resultado = baseline.evaluate(symbol, context)
        baseline.audit_log(symbol, resultado)
        print(f"🔎 Estrategia usada: {baseline.name}")
        print(f"➡️ Decisión tomada: {resultado['action']}")
        print(f"📝 Motivo: {resultado['reason']}")
        return resultado

    async def calculate_reaction_factors(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula chaos_score, strength y reaction_type a partir de los datos de análisis.
        Esta función es asincrónica y puede integrarse en flujos async como analyze().

        Parámetros:
            analysis (dict): Diccionario con resultados previos de análisis.

        Retorna:
            dict: Diccionario con chaos_score, strength y reaction_type.
        """
        try:
            # --- Caos: spread + volatilidad + manipulación ---
            spread_pct = float(analysis.get("spread", {}).get("spread_pct", 0) or 0)
            volatility = float(analysis.get("volatility", 0) or 0)
            manipulation_score = float(analysis.get("anomaly", {}).get("manipulation_risk_score", 0) or 0)

            chaos_score = round(
                min(1.0, spread_pct * 3 + volatility * 2 + manipulation_score),
                3
            )

            # --- Fuerza: sentimiento + señal técnica ---
            sentiment = float(analysis.get("sentiment", 0) or 0)
            signal_strength = float(analysis.get("signal_strength", 0) or 0)

            strength = round(
                min(1.0, sentiment * 0.6 + signal_strength * 0.4),
                3
            )

            # --- Tipo de reacción: depende del ROI y fuerza ---
            roi = float(analysis.get("estimated_roi", 0) or 0)

            if roi > 0.2 and strength > 0.5:
                reaction_type = "positive"
            elif roi < -0.1:
                reaction_type = "inverse"
            else:
                reaction_type = "neutral"

            return {
                "chaos_score": chaos_score,
                "strength": strength,
                "reaction_type": reaction_type
            }

        except Exception as e:
            from krypnova.utils.logger import logger
            logger.error(f"❌ Error calculando reaction_factors: {e}", exc_info=True)
            return {
                "chaos_score": 0.0,
                "strength": 0.0,
                "reaction_type": "unknown"
            }
    async def choose_strategies(
            self,
            profile: str,
            pattern_info: Dict[str, Any],
            vol: float,
            sentiment: float,
            liquidity: float,
            symbol: str
    ) -> List[str]:
        strategies = []
        symbol_lower = symbol.lower()

        # === Análisis de patrones técnicos ===
        patterns = pattern_info.get("patterns", [])
        bullish = any("bullish" in p.lower() for p in patterns)
        bearish = any("bearish" in p.lower() for p in patterns)
        hammer = any("hammer" in p.lower() for p in patterns)
        cup = any("cup and handle" in p.lower() for p in patterns)
        breakout = pattern_info.get("is_breakout", False)
        trap = pattern_info.get("volatility_trap", False)

        if trap:
            strategies.append("trap_reversal")
        if breakout and vol > 0.03:
            strategies.append("htf_breakout")
        if bullish and sentiment > 0.2:
            strategies.append("momentum")
        if bearish and sentiment < -0.2:
            strategies.append("counter_tactic")
        if hammer:
            strategies.append("shadow_strategy")
        if cup:
            strategies.append("phoenix_protocol")

        # === Análisis basado en símbolo ===
        if "meme" in symbol_lower or "pepe" in symbol_lower:
            strategies.append("memecoin_sniper")
        if "usd" in symbol_lower and "bol" in symbol_lower:
            strategies.append("bolivar_mode")
        if "eth" in symbol_lower or "btc" in symbol_lower:
            strategies.append("sniper_mode")

        # === Análisis por perfil ===
        if profile == "aggressive":
            strategies.extend(["phantom_tactics", "hydra_gap", "hydra_fire"])
        elif profile == "moderate":
            strategies.extend(["sniper", "hydra_pulse"])
        elif profile == "defensive":
            strategies.extend(["sentinel", "liquidity_sweeper"])

        # === Filtros adicionales ===
        if liquidity < 10000:
            strategies.append("liquidity_sweeper")

        # === Seguridad adicional para diversidad ===
        if "hydra" not in ",".join(strategies):
            strategies.append("hydra_oracle")

        # Eliminar duplicados
        return list(set(strategies))


    async def suggest_profile(self, user_id: str, window: int = 3) -> str:
        """
        Cambia el perfil de riesgo de manera asíncrona y robusta según los últimos resultados (win/loss).
        - Baja agresividad tras 3 pérdidas consecutivas.
        - Sube agresividad tras 3 ganancias consecutivas.
        - Mantiene si hay resultados mixtos o insuficientes.
        """
        logger = logging.getLogger("ExionProfileAdapt")
        try:
            await self.load_memory()  # Asegura que la memoria esté actualizada
            logs = self.memory.get(user_id, [])
            if not logs or len(logs) < window:
                logger.info(f"[{user_id}] No hay suficientes logs para cambiar perfil (se necesitan {window}).")
                return self.profile  # Mantiene el perfil actual

            last_results = [x.get("result") for x in logs[-window:]]
            count_win = last_results.count("win")
            count_loss = last_results.count("loss")
            current_profile = self.profile

            if count_loss == window:
                if current_profile == "aggressive":
                    new_profile = "moderate"
                elif current_profile == "moderate":
                    new_profile = "conservative"
                else:
                    new_profile = "conservative"
                logger.info(
                    f"🔻 [{user_id}] Cambiando perfil de {current_profile} a {new_profile} (3 pérdidas seguidas).")
                return new_profile

            elif count_win == window:
                if current_profile == "conservative":
                    new_profile = "moderate"
                elif current_profile == "moderate":
                    new_profile = "aggressive"
                else:
                    new_profile = "aggressive"
                logger.info(f"🔺 [{user_id}] Cambiando perfil de {current_profile} a {new_profile} (3 éxitos seguidos).")
                return new_profile

            else:
                logger.info(f"[{user_id}] Resultados mixtos, perfil sin cambios: {current_profile}.")
                return current_profile

        except Exception as e:
            logger.error(f"❌ Error sugiriendo perfil para {user_id}: {e}", exc_info=True)
            return self.profile

    def _normalize_decision(self, decision: str) -> str:
        if not decision:
            return "hold"
        d = decision.strip().lower()
        mapping = {
            "buy": "buy",
            "long": "buy",
            "long_buy": "long_buy",
            "sell": "sell",
            "short": "short",
            "short_sell": "short",
            "scalping": "scalping",
            "limit": "limit_order",
            "limit_order": "limit_order",
            "autoexit": "auto_exit",
            "auto_exit": "auto_exit",
            "stop_loss": "stop_loss",
            "arbitrage": "arbitrage"
        }
        return mapping.get(d, d if d in EXECUTABLE_ACTIONS else "hold")

    async def _get_ohlcv(self, symbol, exchange=None):
        """
        Obtiene el OHLCV para un símbolo (y exchange si se especifica).
        """
        ex = exchange or detect_exchange(symbol)
        df = await self.exchange.get_ohlcv_by_exchange(ex, symbol)
        return df

    def _validate_ohlcv(self, df):
        """
        Valida que el OHLCV tenga la estructura y tamaño mínimo.
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return df is not None and not df.empty and len(df) >= 30 and all(col in df.columns for col in required_cols)

    def _normalize_symbol(self, symbol: str, exchange: str) -> str:
        """
        Normaliza el símbolo según el formato requerido por cada exchange.
        """
        symbol = symbol.upper()
        if exchange == "binance":
            return symbol.replace("/", "").replace("-", "")
        if exchange == "kraken":
            # Kraken usa XBT para BTC
            if symbol in ("BTC/USD", "BTC-USD", "BTCUSD"):
                return "XBTUSD"
            return symbol.replace("/", "")
        if exchange == "coinbase":
            return symbol.replace("/", "-")
        if exchange == "alpaca":
            # Alpaca: solo ticker, sin USD (ej: AAPL)
            if "/" in symbol:
                return symbol.split("/")[0]
            return symbol
        return symbol

    async def get_valid_exchanges_for_symbol(self, symbol: str) -> list:
        valid_exchanges = []
        for exchange in ["kraken", "binance", "coinbase", "alpaca"]:
            try:
                symbols_map = await self.exchange.get_symbols_for_exchange(exchange)
                symbol_norm = self._normalize_symbol(symbol, exchange)
                if isinstance(symbols_map, list) and symbol_norm in symbols_map:
                    valid_exchanges.append(exchange)
                elif isinstance(symbols_map, dict) and symbol_norm in symbols_map.values():
                    valid_exchanges.append(exchange)
            except Exception as e:
                self.logger.warning(f"Error comprobando símbolo en {exchange}: {e}")
                continue
        return valid_exchanges
    async def _get_market_features(self, symbol, df, context):
        """
        Calcula características de mercado: precios, volatilidad, volumen, spread, etc.
        """
        last_price = await get_valid_price(symbol, df)
        volatility = await calculate_volatility(df)
        ewma_volatility = await calculate_ewma_volatility(df)
        average_volume = df["volume"].dropna().mean() if "volume" in df.columns else 1.0
        return {
            "last_price": last_price,
            "volatility": volatility,
            "ewma_volatility": ewma_volatility,
            "average_volume": average_volume,
        }

    async def _get_risk_metrics(self, features, context):
        """
        Calcula métricas de riesgo usando RiskAnalyzer y cuenta.
        Devuelve todas las métricas avanzadas con sus valores.
        """
        import numpy as np
        import logging
        logger = logging.getLogger("RiskMetrics")

        profile = context.get("profile", "moderate")
        capital = context.get("capital", 10000)
        account_data = context.get("account_data", {})

        # Validación robusta de retornos
        returns = features.get("returns")
        if returns is None or not isinstance(returns, (list, np.ndarray)) or len(returns) < 10:
            logger.warning(
                "[RiskMetrics] No hay retornos válidos en features. No se calcularán métricas de riesgo reales.")
            return {  # Retorna todos ceros y logs el problema
                "VaR": 0.0,
                "CVaR": 0.0,
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
                "Max Drawdown": 0.0,
                "Ulcer Index": 0.0,
                "Monte Carlo": {"expected_value": 0.0, "worst_case": 0.0, "best_case": 0.0, "distribution": []},
                "position_size": 0.0,
                "risk_score": 0.0,
                "leverage_eval": "Sin datos",
                "portfolio_health": {},
                "account_risk": {},
            }

        # Si los retornos son todos cero, también alerta
        if np.all(np.array(returns) == 0):
            logger.warning("[RiskMetrics] Todos los retornos son cero. Revisa el cálculo de features['returns'].")

        average_volume = features.get("average_volume", 1.0)
        risk = RiskAnalyzer(
            user_profile=profile,
            capital=capital,
            historical_returns=returns,
            average_volume=average_volume
        )

        # Métricas avanzadas
        var_value = await risk.calculate_var()
        cvar_value = await risk.calculate_cvar()
        sharpe_ratio = await calculate_sharpe_ratio()
        sortino_ratio = await calculate_sortino_ratio()
        max_drawdown = await calculate_max_drawdown()
        ulcer_index = await calculate_ulcer_index()
        monte_carlo = await simulate_monte_carlo(days=30, simulations=1000)
        position_size = await recommend_position_size()
        risk_score = await calculate_risk_score()
        leverage_eval = await evaluate_leverage_limits(current_leverage=account_data.get("leverage", 1.0))
        portfolio_health = await evaluate_portfolio_health()
        account_risk = await assess_account_risk(account_data)

        # Resultado detallado
        return {
            "VaR": var_value,
            "CVaR": cvar_value,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Max Drawdown": max_drawdown,
            "Ulcer Index": ulcer_index,
            "Monte Carlo": monte_carlo,
            "position_size": position_size,
            "risk_score": risk_score,
            "leverage_eval": leverage_eval,
            "portfolio_health": portfolio_health,
            "account_risk": account_risk,
        }

    async def _get_signals(self, symbol: str, features: dict, context: dict) -> dict:
        """
        Calcula señales técnicas e institucionales con robustez y tolerancia a fallos.
        Integra modelos de sentimiento, liquidez, anomalías, orderbook y patrones avanzados.
        Devuelve un dict completo con todos los resultados.
        """
        # === 1️⃣ Modelos de sentimiento, liquidez y anomalías ===
        try:
            sentiment_score = await self.sentiment_model.predict(symbol, context)
        except Exception as e:
            self.logger.error(f"[SIGNALS] ❌ Error en sentiment_model para {symbol}: {e}")
            sentiment_score = 0

        try:
            liquidity = await self.liquidity_model.detect(symbol, context)
        except Exception as e:
            self.logger.error(f"[SIGNALS] ❌ Error en liquidity_model para {symbol}: {e}")
            liquidity = 0

        try:
            anomaly_alert = await self.anomaly_detector.scan(symbol, context)
        except Exception as e:
            self.logger.error(f"[SIGNALS] ❌ Error en anomaly_detector para {symbol}: {e}")
            anomaly_alert = None

        # === 2️⃣ Orderbook con fallback seguro ===
        orderbook_depth = {"bid_ask_ratio": 0, "depth_score": 0}
        spread_info = {"spread": None, "spread_percent": None}

        try:
            orderbook_dict = await self.exchange.get_orderbook(detect_exchange(symbol), symbol)
            if not orderbook_dict or not orderbook_dict.get("bid_price") or not orderbook_dict.get("ask_price"):
                self.logger.warning(f"[SIGNALS] ⚠️ Orderbook vacío o incompleto para {symbol}")
            else:
                orderbook_df = convert_orderbook_json_to_df(orderbook_dict)
                try:
                    orderbook_depth = await self.orderbook_model.measure_depth(symbol, orderbook_df)
                except Exception as e:
                    self.logger.error(f"[SIGNALS] ❌ Error en measure_depth para {symbol}: {e}")
                    orderbook_depth = {"bid_ask_ratio": 0, "depth_score": 0}
                try:
                    spread_info = await self.spread_monitor.check_spread(symbol, orderbook_dict)
                except Exception as e:
                    self.logger.error(f"[SIGNALS] ❌ Error en check_spread para {symbol}: {e}")
                    spread_info = {"spread": None, "spread_percent": None}
        except Exception as e:
            self.logger.error(f"[SIGNALS] ❌ Error obteniendo orderbook para {symbol}: {e}")

        # === 3️⃣ Análisis avanzado de patrones ===
        pattern_info = {
            "patterns": [],
            "is_breakout": False,
            "volatility_trap": False,
            "score": 0,
            "bullish_patterns": [],
            "bearish_patterns": [],
        }
        patterns_score = {"score": 0.0, "volume_score": 0.0, "details": {}}

        try:
            df_ohlcv = features.get("ohlcv")
            if isinstance(df_ohlcv, pd.DataFrame) and not df_ohlcv.empty:
                from pattern_detector import analyze_patterns, evaluate_patterns_score
                # 1️⃣ patrones detectados (bull/bear/lista)
                patterns_detected = await analyze_patterns(df_ohlcv)
                pattern_names = [pat.get("pattern", "").lower() for pat in patterns_detected if pat.get("pattern")]
                bullish_set = {
                    "bullish engulfing", "hammer", "triangle", "cup and handle",
                    "channel up", "wedge up", "morning star", "piercing line"
                }
                bearish_set = {
                    "bearish engulfing", "shooting star", "head and shoulders",
                    "double top", "wedge down", "channel down", "evening star", "dark cloud cover"
                }
                bullish_patterns = [p for p in pattern_names if p in bullish_set]
                bearish_patterns = [p for p in pattern_names if p in bearish_set]
                score = len(bullish_patterns) - len(bearish_patterns)
                is_breakout = any(
                    p in ["triangle", "channel", "flag (bull)", "flag (bear)"] for p in pattern_names
                )
                volatility_trap = any(
                    p in ["head and shoulders", "double top"] for p in pattern_names
                )
                pattern_info = {
                    "patterns": pattern_names,
                    "is_breakout": is_breakout,
                    "volatility_trap": volatility_trap,
                    "score": score,
                    "bullish_patterns": bullish_patterns,
                    "bearish_patterns": bearish_patterns,
                }
                # 2️⃣ Puntaje normalizado y detalles con evaluate_patterns_score
                patterns_score = await evaluate_patterns_score(df_ohlcv)
            else:
                self.logger.warning(
                    f"[SIGNALS] ⚠️ DataFrame OHLCV inválido o vacío para {symbol} en análisis de patrones.")
        except Exception as e:
            self.logger.warning(f"[SIGNALS] ⚠️ Error detectando patrones para {symbol}: {e}")

        # === 4️⃣ Cálculo de fuerza de señal ===
        try:
            signal_strength = self.signal_evaluator.calculate(
                sentiment=sentiment_score,
                volatility=features.get("volatility", 0),
                liquidity=liquidity,
                pattern=pattern_info,
                anomaly=anomaly_alert,
                orderbook_depth=orderbook_depth,
                spread=spread_info,
                patterns_score=patterns_score,
            )
        except Exception as e:
            self.logger.error(f"[SIGNALS] ❌ Error calculando signal_strength para {symbol}: {e}")
            signal_strength = 0

        # === 5️⃣ Retorno completo y robusto ===
        return {
            "sentiment_score": sentiment_score,
            "liquidity": liquidity,
            "anomaly_alert": anomaly_alert,
            "orderbook_depth": orderbook_depth,
            "spread_info": spread_info,
            "pattern_info": pattern_info,
            "patterns_score": patterns_score,
            "signal_strength": signal_strength
        }

    async def _get_drl_predictions(
            self,
            symbol: str,
            df: pd.DataFrame,
            context: dict = None
    ) -> dict:
        """
        Get action predictions from DQN and PPO agents.
        Returns a dict with DQN and PPO predictions, or None if models fail.
        Includes fallback behavior if models are unavailable.
        """
        drl_predictions = {
            "dqn_action": None,
            "dqn_confidence": 0.0,
            "ppo_action": None,
            "ppo_confidence": 0.0,
            "available": False,
            "errors": []
        }

        try:
            # Check if agents are available
            if not hasattr(self, "dqn_agent") or not hasattr(self, "ppo_agent"):
                self.logger.warning(f"[{symbol}] DRL agents not initialized")
                drl_predictions["errors"].append("DRL agents not initialized")
                return drl_predictions

            # Prepare state from dataframe
            # Use last STATE_SIZE features from the dataframe
            if len(df) < STATE_SIZE:
                self.logger.warning(f"[{symbol}] Insufficient data for DRL state (need {STATE_SIZE}, got {len(df)})")
                drl_predictions["errors"].append(f"Insufficient data for DRL state")
                return drl_predictions

            # Extract state features (last STATE_SIZE rows of normalized close prices)
            try:
                close_prices = df["close"].tail(STATE_SIZE).values
                # Normalize the state
                state = (close_prices - np.mean(close_prices)) / (np.std(close_prices) + 1e-8)
                state = state.reshape(1, -1) if len(state.shape) == 1 else state
            except Exception as e:
                self.logger.warning(f"[{symbol}] Error preparing DRL state: {e}")
                drl_predictions["errors"].append(f"State preparation error: {e}")
                return drl_predictions

            # Get DQN prediction
            try:
                if hasattr(self.dqn_agent, "act"):
                    dqn_action = self.dqn_agent.act(state, training=False)
                    # Map action index to decision (0: hold, 1: buy, 2: sell)
                    dqn_action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                    drl_predictions["dqn_action"] = dqn_action_map.get(dqn_action, "HOLD")
                    # Get confidence from Q-values if available
                    if hasattr(self.dqn_agent, "model") and hasattr(self.dqn_agent.model, "predict"):
                        q_values = self.dqn_agent.model.predict(state, verbose=0)
                        max_q = np.max(q_values)
                        mean_q = np.mean(q_values)
                        drl_predictions["dqn_confidence"] = float(max_q / (mean_q + 1e-8)) if mean_q != 0 else 0.5
                    else:
                        drl_predictions["dqn_confidence"] = 0.5
                    self.logger.info(f"[{symbol}] DQN prediction: {drl_predictions['dqn_action']} (confidence: {drl_predictions['dqn_confidence']:.3f})")
            except Exception as e:
                self.logger.warning(f"[{symbol}] DQN prediction failed: {e}")
                drl_predictions["errors"].append(f"DQN error: {e}")

            # Get PPO prediction
            try:
                if hasattr(self.ppo_agent, "act"):
                    ppo_action, ppo_log_prob = self.ppo_agent.act(state)
                    # Map action index to decision (0: hold, 1: buy, 2: sell)
                    ppo_action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                    ppo_action_idx = int(ppo_action) if hasattr(ppo_action, "__int__") else 0
                    drl_predictions["ppo_action"] = ppo_action_map.get(ppo_action_idx, "HOLD")
                    # Use log probability as confidence indicator
                    drl_predictions["ppo_confidence"] = float(np.exp(ppo_log_prob)) if ppo_log_prob is not None else 0.5
                    self.logger.info(f"[{symbol}] PPO prediction: {drl_predictions['ppo_action']} (confidence: {drl_predictions['ppo_confidence']:.3f})")
            except Exception as e:
                self.logger.warning(f"[{symbol}] PPO prediction failed: {e}")
                drl_predictions["errors"].append(f"PPO error: {e}")

            # Mark as available if at least one prediction succeeded
            drl_predictions["available"] = (drl_predictions["dqn_action"] is not None or 
                                           drl_predictions["ppo_action"] is not None)

        except Exception as e:
            self.logger.error(f"[{symbol}] Unexpected error in _get_drl_predictions: {e}", exc_info=True)
            drl_predictions["errors"].append(f"Unexpected error: {e}")

        return drl_predictions

    async def calculate_roi_metrics(
            self,
            symbol: str,
            df: pd.DataFrame,
            capital: float,
            *,
            strategies: list = None,
            context: dict = None,
            trades_df: pd.DataFrame = None,  # para métricas de backtest (profit por trade)
            perf_df: pd.DataFrame = None  # para curva de equity (columna 'total')
    ) -> dict:
        """
        Orquesta señales, riesgo, estrategias y ROI combinado.
        Devuelve un dict con:
          - roi_total, direction, signals_used, detail
          - risk_metrics, signals_info
          - backtest_metrics (Sharpe, win_rate, drawdown, CAGR, etc.)
          - quick_decision (decision, confidence, roi_estimado)
        """
        entry_price = None
        try:
            # 1) Validación y recorte
            if df is None or df.empty or "close" not in df.columns:
                self.logger.warning(f"[{symbol}] ❌ DataFrame inválido o sin 'close'")
                return {
                    "roi_total": 0.0, "direction": "NEUTRAL", "signals_used": [], "detail": {},
                    "risk_metrics": {}, "signals_info": {}, "backtest_metrics": {}, "quick_decision": {},
                    "errors": ["Invalid or empty dataframe."]
                }

            if len(df) > 2000:
                self.logger.warning(f"[{symbol}] ⚠️ DF grande (len={len(df)}), recortando a 1000")
                df = df.tail(1000)

            try:
                entry_price = float(df["close"].iloc[-1])
            except Exception:
                entry_price = None

            # 2) Señales base (sync)
            feature_dict = {
                "ohlcv": df,
                "atr": float(df["atr"].iloc[-1]) if "atr" in df.columns and not df.empty else 0.01,
                "volatility": float(df["volatility"].iloc[-1]) if "volatility" in df.columns and not df.empty else 0.01,
                "liquidity_score": float(
                    df["liquidity_score"].iloc[-1]) if "liquidity_score" in df.columns and not df.empty else None,
                "rsi": float(df["rsi"].iloc[-1]) if "rsi" in df.columns and not df.empty else None,
                "macd": float(df["macd_line"].iloc[-1]) if "macd_line" in df.columns and not df.empty else None,
                "macd_signal": float(
                    df["signal_line"].iloc[-1]) if "signal_line" in df.columns and not df.empty else None,
            }
            try:
                signals_info = await self._get_signals(symbol, feature_dict, context or {})
            except Exception as e:
                self.logger.warning(f"[{symbol}] ⚠️ _get_signals: {e}")
                signals_info = {}

            # 3) Métricas de riesgo (async)
            try:
                risk_metrics = await self._get_risk_metrics({
                    "returns": feature_dict.get("ohlcv")["close"].pct_change().dropna().tolist(),
                    "average_volume": float(df["volume"].tail(50).mean()) if "volume" in df.columns else 1.0,
                }, context or {"profile": "moderate", "capital": capital})
            except Exception as e:
                self.logger.warning(f"[{symbol}] ⚠️ _get_risk_metrics: {e}")
                risk_metrics = {}

                # --- 4) Selección e instanciación de estrategias válidas ---
                built = []
                errors = []

                if not strategies or not (hasattr(strategies[0], "calculate_metrics") if strategies else False):
                    try:
                        # Usa la instancia persistente si existe, sino créala
                        if not self.offensive_selector:
                            self.offensive_selector = OffensiveSelector(
                                user_profile=(context or {}).get("profile", "moderate"),
                                user_id=(context or {}).get("user_id", None)
                            )
                        # Obtén estrategias sugeridas (async)
                        strategy_names = await self.offensive_selector.prod_dynamic_strategy_selector(symbol,
                                                                                                      context or {})
                        if not strategy_names:
                            self.logger.warning(f"[{symbol}] No hay estrategias sugeridas. Fallback a buy_and_hold.")
                            strategy_names = ["buy_and_hold"]
                        strategy_classes = [
                            self.offensive_selector.available_strategies.get(name)
                            for name in strategy_names
                            if name in self.offensive_selector.available_strategies
                        ]
                    except Exception as e:
                        self.logger.warning(f"[{symbol}] Error al seleccionar estrategias: {e}")
                        errors.append(str(e))
                        strategy_classes = []

                    # Instancia las estrategias sugeridas
                    for Strat in strategy_classes:
                        try:
                            inst = _build_strategy_instance(Strat, context, capital, symbol)
                            if inspect.isawaitable(inst):
                                inst = await inst
                            if isinstance(inst, Exception) or inst is None:
                                continue
                            built.append(inst)
                        except Exception as e:
                            self.logger.warning(f"[Factory] No instanciable {Strat}: {e}")
                            errors.append(f"{Strat}: {e}")

                    strategies = built

                # --- Fallback: si ningún build funcionó, muestra disponibles y fuerza una ---
                if not strategies:
                    available_strat_names = list(getattr(self.offensive_selector, "available_strategies", {}).keys())
                    self.logger.warning(
                        f"[{symbol}] ⚠️ Sin estrategias válidas instanciadas. Estrategias disponibles: {available_strat_names}"
                    )
                    # Fallback: usa buy_and_hold si existe, si no la primera disponible
                    fallback_name = "buy_and_hold" if "buy_and_hold" in available_strat_names else (
                        available_strat_names[0] if available_strat_names else None)
                    if fallback_name:
                        Strat = self.offensive_selector.available_strategies[fallback_name]
                        try:
                            inst = _build_strategy_instance(Strat, context, capital, symbol)
                            if inspect.isawaitable(inst):
                                inst = await inst
                            if inst and not isinstance(inst, Exception):
                                strategies = [inst]
                                self.logger.warning(f"[{symbol}] Fallback a estrategia: {fallback_name}")
                            else:
                                self.logger.error(f"[{symbol}] Fallback instanciado pero inválido: {fallback_name}")
                                strategies = []
                        except Exception as e:
                            self.logger.error(f"[{symbol}] Error instanciando fallback {fallback_name}: {e}")
                            strategies = []
                    else:
                        self.logger.error(f"[{symbol}] No hay ninguna estrategia disponible para fallback.")
                        strategies = []

                # --- Si AÚN no hay estrategias, retorno neutro. Si hay, ejecuta y registra la seleccionada ---
                if not strategies:
                    self.logger.warning(f"[{symbol}] ⚠️ Sin estrategias disponibles tras fallback; retorno neutro")
                    combined_result = {"roi_total": 0.0, "direction": "NEUTRAL", "signals_used": [], "detail": {},
                                       "strategy": "N/A"}
                else:
                    # Ejecuta la(s) estrategia(s) y registra nombre y ROI
                    signals = []
                    for strat in strategies:
                        try:
                            roi, mc_roi, metrics, mc_results = await _safe_calculate_metrics(strat, symbol, df, capital)
                            action, confidence = await _safe_action_and_confidence(strat, df, metrics, mc_results)
                            signals.append({
                                "name": getattr(strat, "name", strat.__class__.__name__),
                                "action": action,
                                "roi": roi,
                                "confidence": confidence
                            })
                        except Exception as e:
                            self.logger.warning(f"Error ejecutando {strat}: {e}")
                            errors.append(str(e))
                    if not signals:
                        self.logger.warning(f"[{symbol}] ⚠️ Todas las estrategias fallaron o devolvieron None.")
                        combined_result = {"roi_total": 0.0, "direction": "NEUTRAL", "signals_used": [], "detail": {},
                                           "strategy": "N/A"}
                    else:
                        # Elige la mejor señal (ejemplo: mayor ROI)
                        best_signal = max(signals, key=lambda x: x.get("roi") or 0)
                        combined_result = {
                            "roi_total": best_signal["roi"],
                            "direction": best_signal["action"].upper(),
                            "signals_used": signals,
                            "detail": {},
                            "strategy": best_signal["name"]
                        }
                # 5) Ejecutar estrategias en paralelo
                async def run_one(strat):
                    name = getattr(strat, "name", strat.__class__.__name__)
                    try:
                        roi, mc_roi, metrics, mc_results = await asyncio.wait_for(
                            _safe_calculate_metrics(strat, symbol, df, capital), timeout=10
                        )
                        action, confidence = await _safe_action_and_confidence(strat, df, metrics, mc_results)
                        return {"name": name, "action": action, "roi": roi, "confidence": confidence}
                    except asyncio.TimeoutError:
                        self.logger.warning(f"[{symbol}:{name}] ⚠️ Timeout")
                        errors.append(f"{name}: Timeout")
                    except Exception as e:
                        self.logger.warning(f"[{symbol}:{name}] ⚠️ Error: {e}")
                        errors.append(f"{name}: {e}")
                    return None

                tasks = [run_one(s) for s in strategies]
                results = await asyncio.gather(*tasks)
                signals = [s for s in results if s and s.get("name")]

                if not signals:
                    self.logger.warning(f"[{symbol}] ⚠️ Todas las estrategias fallaron o devolvieron None.")
                    combined_result = {"roi_total": 0.0, "direction": "NEUTRAL", "signals_used": [], "detail": {}}
                else:
                    cap_float = float(capital) if isinstance(capital, (int, float, str)) else 0.0
                    capital_dict = {s["name"]: cap_float for s in signals}
                    try:
                        combined_raw = combine_signals_roi(signals, capital_dict)
                        combined_result = await _maybe_await(combined_raw)
                        if not isinstance(combined_result, dict):
                            combined_result = {"roi_total": 0.0, "direction": "NEUTRAL", "signals_used": [],
                                               "detail": {}}
                    except Exception as e:
                        self.logger.warning(f"[{symbol}] Error combinando señales: {e}")
                        combined_result = {"roi_total": 0.0, "direction": "NEUTRAL", "signals_used": [], "detail": {}}
                        errors.append(f"combine_signals_roi: {e}")

            # 5.5) Get DRL predictions (DQN and PPO) and integrate with strategy outputs
            drl_predictions = await self._get_drl_predictions(symbol, df, context)
            
            # Log DRL contributions for debugging
            if drl_predictions.get("available"):
                self.logger.info(
                    f"[{symbol}] DRL Predictions - DQN: {drl_predictions.get('dqn_action')} "
                    f"(conf: {drl_predictions.get('dqn_confidence', 0):.3f}), "
                    f"PPO: {drl_predictions.get('ppo_action')} "
                    f"(conf: {drl_predictions.get('ppo_confidence', 0):.3f})"
                )
                
                # Integrate DRL predictions with combined_result
                # Add DRL predictions as additional signals
                if drl_predictions.get("dqn_action"):
                    signals.append({
                        "name": "DQNAgent",
                        "action": drl_predictions["dqn_action"],
                        "roi": 0.0,  # DRL doesn't directly provide ROI
                        "confidence": drl_predictions["dqn_confidence"]
                    })
                
                if drl_predictions.get("ppo_action"):
                    signals.append({
                        "name": "PPOAgent",
                        "action": drl_predictions["ppo_action"],
                        "roi": 0.0,  # DRL doesn't directly provide ROI
                        "confidence": drl_predictions["ppo_confidence"]
                    })
                
                # Recalculate combined result with DRL predictions
                if signals:  # Only if we have any signals (including DRL)
                    # Calculate consensus direction from all signals
                    action_votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0, "LONG": 0.0, "SHORT": 0.0}
                    total_confidence = 0.0
                    
                    for sig in signals:
                        action = sig.get("action", "HOLD").upper()
                        confidence = sig.get("confidence", 0.5)
                        # Normalize action names
                        if action in ["BUY", "LONG"]:
                            action_votes["BUY"] += confidence
                        elif action in ["SELL", "SHORT"]:
                            action_votes["SELL"] += confidence
                        else:
                            action_votes["HOLD"] += confidence
                        total_confidence += confidence
                    
                    # Determine consensus direction
                    consensus_action = max(action_votes.items(), key=lambda x: x[1])[0]
                    
                    # Update combined_result with DRL-enhanced direction
                    original_direction = combined_result.get("direction", "NEUTRAL")
                    combined_result["direction_with_drl"] = consensus_action
                    combined_result["drl_consensus_strength"] = action_votes[consensus_action] / (total_confidence + 1e-8)
                    
                    # Update signals_used to include DRL
                    combined_result["signals_used"] = signals
                    
                    # Add DRL detail to output
                    combined_result["detail"]["drl_predictions"] = {
                        "dqn": {
                            "action": drl_predictions.get("dqn_action"),
                            "confidence": drl_predictions.get("dqn_confidence", 0.0)
                        },
                        "ppo": {
                            "action": drl_predictions.get("ppo_action"),
                            "confidence": drl_predictions.get("ppo_confidence", 0.0)
                        },
                        "original_direction": original_direction,
                        "enhanced_direction": consensus_action,
                        "action_votes": action_votes
                    }
                    
                    self.logger.info(
                        f"[{symbol}] DRL Integration - Original: {original_direction}, "
                        f"DRL-Enhanced: {consensus_action}, Consensus strength: {combined_result['drl_consensus_strength']:.3f}"
                    )
            else:
                # Fallback: No DRL predictions available
                self.logger.info(f"[{symbol}] DRL predictions not available, using strategy-only predictions")
                combined_result["detail"]["drl_predictions"] = {
                    "available": False,
                    "errors": drl_predictions.get("errors", [])
                }

            # 6) Métricas de backtest (Sharpe, win-rate, drawdown, CAGR, etc.)
            try:
                backtest_metrics = await compile_backtest_metrics(perf_df or pd.DataFrame(),
                                                                  trades_df or pd.DataFrame())
            except Exception as e:
                self.logger.warning(f"[{symbol}] ⚠️ Backtest metrics: {e}")
                backtest_metrics = {}
                errors.append(f"backtest_metrics: {e}")

            # 7) Decisión rápida basada en SMA/RSI (perfil del usuario)
            try:
                profile = (context or {}).get("profile", "moderate")
                decision, confidence, roi_est = calcula_decision_y_confianza(df.copy(), profile=profile)
                quick_decision = {"decision": decision, "confidence": confidence, "roi_estimate": roi_est}
            except Exception as e:
                self.logger.warning(f"[{symbol}] ⚠️ Quick decision: {e}")
                quick_decision = {}
                errors.append(f"quick_decision: {e}")

            # 8) Ensamble final
            output = {
                "roi_total": combined_result.get("roi_total", 0.0),
                "direction": combined_result.get("direction", "NEUTRAL"),
                "direction_with_drl": combined_result.get("direction_with_drl", combined_result.get("direction", "NEUTRAL")),
                "drl_consensus_strength": combined_result.get("drl_consensus_strength", 0.0),
                "signals_used": combined_result.get("signals_used", []),
                "detail": combined_result.get("detail", {}),
                "risk_metrics": risk_metrics,
                "signals_info": signals_info,
                "backtest_metrics": backtest_metrics,  # usa Sharpe/win/drawdown/CAGR de tu archivo
                "quick_decision": quick_decision,  # decisión complementaria (SMA/RSI)
            }
            if errors:
                output["errors"] = errors
            # Logging resumido para producción
            self.logger.info(f"[{symbol}] OUTPUT SUMMARY: {output}")
            return output

        except Exception as e:
            self.logger.warning(
                f"[{symbol}] ❌ Error inesperado en calculate_roi_metrics: {e} | entry_price: {entry_price}\n{traceback.format_exc()}"
            )
            return {
                "roi_total": 0.0, "direction": "NEUTRAL", "signals_used": [], "detail": {},
                "risk_metrics": {}, "signals_info": {}, "backtest_metrics": {}, "quick_decision": {},
                "errors": [f"unexpected: {e}"]
            }

    async def analyze_daily_opportunities(self, context: dict, top_n: int = 5) -> list:
        """
        Analiza oportunidades de trading considerando:
          - Favoritos del usuario (si existen)
          - Los mejores movers del día (gainers/losers) que NO estén entre los favoritos
        Ejecuta el pipeline y sugiere los TOP N con mejor ROI x Sharpe ratio.
        """
        favoritos = set(context.get("favoritos", []))
        results = []

        # 1. Obtener top movers del día desde todos los exchanges
        mover_fetcher = TopMoversFetcher()
        all_movers = await mover_fetcher.get_all_movers(limit=10)
        movers_symbols = set()
        for movers in all_movers:
            movers_symbols.update([s for s, _ in movers.get("gainers", [])])
            movers_symbols.update([s for s, _ in movers.get("losers", [])])

        # 2. Lista: favoritos + movers no favoritos (sin duplicados)
        symbols_to_analyze = list(favoritos) + [s for s in movers_symbols if s not in favoritos]

        # 3. Análisis
        for symbol in symbols_to_analyze:
            try:
                analysis = await self.analyze(symbol, context)
                for res in analysis:
                    roi = res.get("roi")
                    sharpe = res.get("portfolio_metrics", {}).get("sharpe_ratio")
                    # Solo toma buy/sell con ambos valores no nulos y positivos
                    if (
                            res.get("decision") in ["buy", "sell"] and
                            roi is not None and roi > 0 and
                            sharpe is not None and sharpe > 0
                    ):
                        # Métrica combinada ROI * Sharpe (puedes ajustar el peso)
                        res["roi_sharpe_score"] = roi * sharpe
                        results.append(res)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.warning(f"Error analizando {symbol}: {e}")

        # 4. Ordenar por ROI x Sharpe (descendente)
        results = sorted(
            results,
            key=lambda x: x.get("roi_sharpe_score", 0),
            reverse=True
        )

        # 5. Devolver solo los top_n mejores (por defecto 5)
        top_results = results[:top_n]

        # 6. (Opcional) ALERTA: aquí puedes agregar lógica para enviar notificaciones
        # for opp in top_results:
        #     await self.send_alert(opp)

        return top_results

    async def analyze(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # --- Defaults y estado ---
        estimated_roi = 0.0
        montecarlo_summary: Dict[str, Any] = {}
        backtest_summary: Dict[str, Any] = {}
        position_size = 0.0
        risk_score = 0
        leverage_eval = {}
        account_risk = {}
        decision = "hold"  # usar minúsculas internamente
        confidence = 0.5
        strategy_mode = "none"
        last_price = None
        sentiment_score = 0.0
        volatility = 0.0
        ewma_volatility = 0.0
        liquidity = 0.0
        anomaly_alert: Dict[str, Any] = {}
        orderbook_depth: Dict[str, Any] = {}
        spread_info: Dict[str, Any] = {}
        pattern_info: Dict[str, Any] = {"patterns": [], "is_breakout": False, "volatility_trap": False}
        risk_params: Dict[str, Any] = {}
        montecarlo_result = None

        # Parámetros de riesgo que podrían no definirse si hay error
        stop_loss = None
        take_profit = None
        trailing_stop = None
        rr_ratio = None

        # Perfil dinámico
        user_id = context.get("user_id", getattr(self, "user_id", "default_user"))
        self.profile = await self.suggest_profile(user_id)
        context["profile"] = self.profile

        try:
            logger.info(f"🧠 Analizando {symbol} en contexto avanzado...")
            exchange_name = detect_exchange(symbol)
            resolved_symbol = await self.exchange.resolve_symbol(exchange_name, symbol)
            df = await self.exchange.get_ohlcv_by_exchange(exchange_name, resolved_symbol)

            logger.info(
                f"🧩 Exchange detectado: {exchange_name} | Símbolo original: {symbol} → Usando: {resolved_symbol}")

            # Validación OHLCV
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if df is None or df.empty or len(df) < 30 or not all(c in df.columns for c in required_cols):
                logger.error(f"[{exchange_name}] ❌ OHLCV insuficiente o columnas faltantes: {required_cols}")
                return {"symbol": symbol, "error": "Sin datos OHLCV válidos", "mode": "none"}

            # Precio válido
            last_price = await get_valid_price(resolved_symbol, df)
            if last_price is None or last_price <= 0.0:
                logger.error(f"[{resolved_symbol}] ⚠️ Precio final inválido: {last_price}")
                return {"symbol": symbol, "error": "Precio no disponible", "mode": "none"}

            profile = context.get("profile", "moderate")
            capital = context.get("capital", 10000)
            account_data = context.get("account_data", {})

            # --- INTEGRA calculate_roi_metrics aquí ---
            roi_metrics = await self.calculate_roi_metrics(
                resolved_symbol, df, capital, context=context
            )
            # Extrae los datos clave
            estimated_roi = roi_metrics.get("roi_total", 0.0)
            direction = roi_metrics.get("direction", "NEUTRAL")
            quick_decision = roi_metrics.get("quick_decision", {})
            backtest_summary = roi_metrics.get("backtest_metrics", {})
            signals_info = roi_metrics.get("signals_info", {})
            risk_metrics = roi_metrics.get("risk_metrics", {})
            signals_used = roi_metrics.get("signals_used", [])
            errors = roi_metrics.get("errors", [])

            # Defensa
            try:
                prices = df["close"].dropna().tolist()
                defense_result = await self.defense.defense_decision(exchange_name, resolved_symbol, df, prices)
                logger.info(f"[{resolved_symbol}] 🏯️ Defensa: {defense_result}")
                if isinstance(defense_result, dict):
                    strategy_mode = "defensive" if defense_result.get("decision") in ["WAIT", "HOLD"] else "analyzing"
                elif isinstance(defense_result, str):
                    strategy_mode = "defensive" if defense_result.lower() == "block" else "analyzing"
                else:
                    strategy_mode = "analyzing"
            except Exception as e:
                logger.warning(f"⚠️ Fallo ExionDefense: {e}")
                score = await self.defense.compute_abnormal_behavior_score(resolved_symbol)
                strategy_mode = "defensive" if score > 0.7 else "analyzing"

            # Retornos / volumen
            try:
                returns = df["close"].pct_change().dropna().tolist() or [0.0] * 30
            except Exception:
                returns = [0.0] * 30
            try:
                average_volume = df["volume"].dropna().mean() if "volume" in df.columns else 1.0
                if not np.isfinite(average_volume) or average_volume == 0:
                    average_volume = 1.0
            except Exception:
                average_volume = 1.0

            # Risk Analyzer
            try:
                risk = RiskAnalyzer(
                    user_profile=profile,
                    capital=capital,
                    historical_returns=returns,
                    average_volume=average_volume
                )
                portfolio_health = await risk.evaluate_portfolio_health()
                position_size = await risk.recommend_position_size()
                risk_score = await risk.calculate_risk_score()
                leverage_eval = await risk.evaluate_leverage_limits(current_leverage=account_data.get("leverage", 1.0))
            except Exception as e:
                logger.warning(f"⚠️ RiskAnalyzer: {e}")
                portfolio_health, position_size, risk_score, leverage_eval = {}, 0.0, 0, {}

            # Riesgo de cuenta
            try:
                account_risk = await assess_account_risk(account_data)
                if account_risk.get("global_risk_exposure", 0) > 80:
                    return {
                        "symbol": symbol,
                        "decision": "wait",
                        "confidence": 0.3,
                        "price": last_price,
                        "reason": "❌ Riesgo de cuenta elevado. Evitar nuevas posiciones.",
                        "risk_alert": account_risk,
                        "estimated_roi": 0.0,
                        "montecarlo": {},
                        "backtest": {}
                    }
            except Exception as e:
                logger.warning(f"⚠️ Evaluación de cuenta fallida: {e}")

            # Datos extra / modelos
            try:
                volatility = await calculate_volatility(df)
                ewma_volatility = await calculate_ewma_volatility(df)
                market_context = await self.context_analyzer.evaluate(resolved_symbol, context)
                sentiment_score = await self.sentiment_model.predict(resolved_symbol, context)
                liquidity = await self.liquidity_model.detect(resolved_symbol, context)
                anomaly_alert = await self.anomaly_detector.scan(resolved_symbol, context)

                orderbook_df = await self.exchange.get_orderbook_df(exchange_name, resolved_symbol)
                orderbook_depth = await self.orderbook_model.measure_depth(resolved_symbol, orderbook_df)
                spread_info = await self.spread_monitor.check_spread(resolved_symbol)

                try:
                    if isinstance(orderbook_df, pd.DataFrame) and not orderbook_df.empty:
                        snap = orderbook_df.iloc[0].to_dict()
                        orderbook_snapshot = {
                            "bid": float(snap.get("bid_price", 0.0)),
                            "ask": float(snap.get("ask_price", 0.0)),
                            "bid_volume": float(snap.get("bid_volume", 0.0)),
                            "ask_volume": float(snap.get("ask_volume", 0.0)),
                        }
                        bot_activity = await self.networking_signal.detect_bot_behavior(orderbook_snapshot)
                        bait_trigger_mode = await self.networking_signal.bait_and_trigger(resolved_symbol, last_price,
                                                                                          volatility)
                        if bot_activity:
                            strategy_mode = "defensive"
                        elif bait_trigger_mode in ["bait", "trigger"]:
                            strategy_mode = bait_trigger_mode
                            # demo arbitraje simulado (protegido por try/except interno)
                            price_feeds = {
                                "BINANCE": last_price,
                                "KRAKEN": last_price + random.uniform(-0.3, 0.3),
                                "COINBASE": last_price + random.uniform(-0.2, 0.2),
                                "ALPACA": last_price + random.uniform(-0.1, 0.1),
                            }
                            buy_ex, buy_px = min(price_feeds.items(), key=lambda x: x[1])
                            sell_ex, sell_px = max(price_feeds.items(), key=lambda x: x[1])
                            profit = sell_px - buy_px
                            try:
                                if await self.networking_signal.simulate_latency_advantage(buy_px,
                                                                                           sell_px) and profit > 0:
                                    strategy_mode = "arbitrage"
                                    await self.arbitrage_system.execute_arbitrage(
                                        resolved_symbol, buy_exchange=buy_ex, buy_price=buy_px,
                                        sell_exchange=sell_ex, sell_price=sell_px, profit=profit
                                    )
                            except Exception as e:
                                logger.warning(f"⚠️ Arbitrage sim error {resolved_symbol}: {e}")
                except Exception as e:
                    logger.warning(f"⚠️ Networking signal error {resolved_symbol}: {e}")
            except Exception as e:
                logger.warning(f"⚠️ Datos extra/modelos error {resolved_symbol}: {e}")

            # Análisis técnico
            atr = 0.01
            try:
                if len(df) >= 20:
                    detected = await analyze_patterns(df)
                    names = [p.get("pattern", "").lower() for p in detected]
                    pattern_info = {
                        "patterns": names,
                        "is_breakout": any(p in names for p in ["triangle", "channel", "flag (bull)", "flag (bear)"]),
                        "volatility_trap": any(p in names for p in ["head and shoulders", "double top"]),
                    }
                    atr_series = await calculate_atr(df)
                    atr_val = float(atr_series.iloc[-1]) if isinstance(atr_series, pd.Series) else float(atr_series)
                    atr = atr_val if np.isfinite(atr_val) and atr_val > 0 else 0.01
            except Exception as e:
                logger.warning(f"⚠️ Patrones/ATR error: {e}")

            # PatternAttackExecutor (no bloqueante)
            try:
                _ = await self.pattern_executor.analyze_and_attack(resolved_symbol, df, last_price)
            except Exception as e:
                logger.warning(f"[{resolved_symbol}] PatternAttackExecutor error: {e}")

            # MonteCarlo
            try:
                log_returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
                if len(log_returns) > 0 and np.isfinite(log_returns).all():
                    mu, sigma = float(np.mean(log_returns)), float(np.std(log_returns) or 1e-6)
                    simulator = MonteCarloAsyncSimulator(resolved_symbol, last_price, mu, sigma, 252, 1000)
                    montecarlo_result = await simulator.run_simulation_async()
                    if isinstance(montecarlo_result, np.ndarray) and montecarlo_result.shape[1] > 0:
                        finals = montecarlo_result[:, -1]
                        estimated_roi = float(np.mean((finals - last_price) / last_price) * 100.0)  # %
                        montecarlo_summary = {
                            "final_prices_mean": float(np.mean(finals)),
                            "final_prices_std": float(np.std(finals)),
                            "final_prices_min": float(np.min(finals)),
                            "final_prices_max": float(np.max(finals)),
                            "roi_mean_percent": estimated_roi
                        }
            except Exception as e:
                logger.error(f"MonteCarlo error {resolved_symbol}: {e}", exc_info=True)
                montecarlo_summary = {"error": str(e)}

            # Backtest (proteger firma)
            try:
                if "strategy" in context:
                    bt = await backtest(
                        symbols_by_exchange={exchange_name: [resolved_symbol]},
                        capital=capital, profile=profile, strategy=context.get("strategy", "bolivar_mode")
                    )
                else:
                    # fallback si tu backtest espera df/strategy_name
                    bt = await backtest(df, capital, profile, "bolivar_mode")
                backtest_summary = (
                    bt.to_dict() if hasattr(bt, "to_dict") else
                    (bt if isinstance(bt, dict) else {"result": str(bt)})
                )
            except Exception as e:
                logger.error(f"Backtest error {resolved_symbol}: {e}", exc_info=True)
                backtest_summary = {"error": str(e)}

            # Señal + riesgo
            try:
                signal_strength = self.signal_evaluator.calculate(
                    sentiment=sentiment_score,
                    volatility=volatility,
                    liquidity=liquidity,
                    pattern=pattern_info,
                    anomaly=anomaly_alert,
                    orderbook_depth=orderbook_depth,
                    spread=spread_info
                )
                stop_loss = await calculate_dynamic_stop_loss(last_price, volatility, context)
                take_profit = await calculate_dynamic_take_profit(
                    price=last_price, atr=atr, strength=signal_strength, sentiment=sentiment_score, profile=profile
                )
                trailing_stop = await calculate_trailing_stop(last_price, last_price, atr, profile)
                risk_amount = await max_risk_per_trade(capital, profile)
                position_size = await calculate_position_size(capital, last_price, stop_loss, profile)
                trade_risk = await evaluate_trade_risk(
                    entry_price=last_price, atr=atr, profile=profile, capital=capital, sentiment=sentiment_score
                )
                rr_ratio = await risk_reward_ratio(last_price, stop_loss, take_profit)

                risk_params = {
                    "entry": last_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trailing_stop": trailing_stop,
                    "position_size": position_size,
                    "trade_risk": trade_risk,
                    "account_risk": account_risk,
                    "risk_reward_ratio": rr_ratio,
                }
            except Exception as e:
                logger.warning(f"⚠️ Parámetros de riesgo error: {e}")

            # Factores de reacción (resiliente)
            analysis = {
                "symbol": resolved_symbol,
                "sentiment": sentiment_score,
                "volatility": volatility,
                "signal_strength": signal_strength if 'signal_strength' in locals() else 0.0,
                "liquidity": liquidity,
                "anomaly": anomaly_alert,
                "pattern": pattern_info,
                "spread": spread_info,
                "decision": decision,
                "estimated_roi": estimated_roi,
                "risk_score": risk_score
            }
            try:
                reaction_factors = await calculate_reaction_factors(analysis)
                analysis.update(reaction_factors)
            except Exception as e:
                logger.warning(f"⚠️ Factores de reacción error: {e}")
                analysis.update({
                    "chaos_score": round(random.uniform(0.2, 0.6), 2),
                    "strength": round(random.uniform(0.1, 0.4), 2),
                    "reaction_type": "neutral"
                })

            # Umbrales por perfil (en %)
            min_roi_threshold = 1.0
            max_risk_allowed = 85
            min_mc_roi = 1.0
            if profile == "conservative":
                min_roi_threshold, max_risk_allowed, min_mc_roi = 2.0, 70, 2.0
            elif profile == "aggressive":
                min_roi_threshold, max_risk_allowed, min_mc_roi = 0.1, 95, 0.1

            mc_roi = float(montecarlo_summary.get("roi_mean_percent", 0.0))
            trend = backtest_summary.get("trend")
            if trend is None:
                try:
                    closes = df["close"].tail(60).values
                    trend = "up" if np.polyfit(np.arange(len(closes)), closes, 1)[0] > 0 else "down"
                except Exception:
                    trend = "down"

            # --- INTEGRADOR DE DECISIÓN FINAL: prioriza señales sobre heurística ---

            DIRECCIONES_ACCIONES = {
                "LONG": "buy",
                "BUY": "buy",
                "SHORT": "sell",
                "SELL": "sell",
                "SCALPING": "scalping",
                "ARBITRAGE": "arbitrage",
                "STOP_LOSS": "stop_loss",
                "TAKE_PROFIT": "take_profit"
            }

            # Si la defensa bloquea/espera
            if isinstance(defense_result, dict) and defense_result.get("decision", "").upper() in ["BLOCK", "WAIT",
                                                                                                   "HOLD"]:
                decision = "hold"
                reason = f"Defensa bloquea/espera: {defense_result.get('reason', 'Riesgo detectado')}"
                strategy_mode = "defensive"

            # Si dirección de señales es clara y fuerza >= umbral, o bien alguna acción especial
            elif direction and direction.upper() in DIRECCIONES_ACCIONES and (
                    (direction.upper() in ["LONG", "BUY"] and roi_metrics.get("detail", {}).get("long_strength",
                                                                                                0) >= 0.5) or
                    (direction.upper() in ["SHORT", "SELL"] and roi_metrics.get("detail", {}).get("short_strength",
                                                                                                  0) >= 0.5) or
                    (direction.upper() in ["SCALPING", "ARBITRAGE", "STOP_LOSS", "TAKE_PROFIT"])
            ):
                decision = DIRECCIONES_ACCIONES[direction.upper()]
                reason = f"Señal ofensiva fuerte: {direction.upper()}"
                strategy_mode = (roi_metrics.get("signals_used", [{}])[0].get("name", "signals")
                                 if roi_metrics.get("signals_used") else "signals")

            # Si ROI muy bueno (aunque fuerza baja), permite acción
            elif estimated_roi > 0.08:
                decision = "buy"
                reason = f"ROI positivo aunque fuerza baja: {estimated_roi:.3f}"
                strategy_mode = (roi_metrics.get("signals_used", [{}])[0].get("name", "signals")
                                 if roi_metrics.get("signals_used") else "signals")
            elif estimated_roi < -0.08:
                decision = "sell"
                reason = f"ROI negativo aunque fuerza baja: {estimated_roi:.3f}"
                strategy_mode = (roi_metrics.get("signals_used", [{}])[0].get("name", "signals")
                                 if roi_metrics.get("signals_used") else "signals")

            # Fallback a lógica heurística/quick_decision
            else:
                # Aquí puedes conservar tu heurística previa como fallback, o dejar sólo 'hold'
                decision = "hold"
                reason = f"Condiciones neutras. Perfil {profile}"
            # Mínimo tamaño si buy/short
            if position_size == 0 and decision in ["buy", "short"]:
                position_size = capital * 0.01 / last_price

            # Memoria/portfolio (resiliente)
            try:
                portfolio = PortfolioManager(user_id=getattr(self, "user_id", "default_user"))
                await portfolio.update(resolved_symbol, decision, position_size, last_price, estimated_roi)
                prev = self.symbol_memory.get(resolved_symbol, {"roi": 0.0, "decision": ""})
                if abs(prev.get("roi", 0.0) - estimated_roi) > 10:
                    logger.info(f"[{resolved_symbol}] 🧠 Cambio brusco ROI: {prev.get('roi')} → {estimated_roi}")
                self.symbol_memory[resolved_symbol] = {"roi": estimated_roi, "decision": decision, "risk": risk_score}
                if prev.get("decision") == decision:
                    logger.info(f"[{resolved_symbol}] 🔁 Repite decisión: {decision.upper()}")
            except Exception as e:
                logger.warning(f"⚠️ Portfolio/memory error: {e}")

            # Ataque económico si defensa dejó WAIT/HOLD pero hay oportunidad
            econ = {
                "conservative": {"min_roi": 1.0, "max_risk": 75, "min_signal": 0.5},
                "moderate": {"min_roi": 0.5, "max_risk": 85, "min_signal": 0.3},
                "aggressive": {"min_roi": 0.1, "max_risk": 95, "min_signal": 0.1},
            }.get(profile, {"min_roi": 0.5, "max_risk": 85, "min_signal": 0.3})

            if decision in ["wait", "hold"] and estimated_roi > econ["min_roi"] and risk_score < econ["max_risk"] and (
                    analysis.get("strength", 0) > econ["min_signal"]):
                decision = "buy"
                reason = f"📈 Oportunidad ofensiva ({profile}) tras defensa pasiva"

            # --- Único return final ---
            return {
                "symbol": resolved_symbol,
                "decision": decision,
                "confidence": confidence,
                "price": last_price,
                "reason": reason,
                "estimated_roi": estimated_roi,
                "montecarlo": montecarlo_summary,
                "backtest": backtest_summary,
                "volatility": volatility,
                "spread": spread_info,
                "anomaly": anomaly_alert,
                "sentiment": sentiment_score,
                "signal_strength": analysis.get("strength", 0.0),
                "position_size": position_size,
                "entry_price": last_price,
                "exit_price": take_profit,
                "risk_parameters": risk_params,
                "risk_score": risk_score,
                "account_risk_score": account_risk,
                "chaos_score": analysis.get("chaos_score", 0.0),
                "reaction_type": analysis.get("reaction_type", "neutral"),
                "mode": strategy_mode
            }

        except Exception as e:
            logger.error(f"❌ Error en análisis de {symbol}: {e}", exc_info=True)
            return {
                "symbol": symbol,
                "error": str(e),
                "mode": strategy_mode,
                "montecarlo": montecarlo_summary,
                "backtest": backtest_summary,
                "account_risk_score": account_risk
            }
    async def analyze_market_institutional(
            self,
            symbols: list,
            exchanges: list = None,
            context: dict = None,
            concurrency: int = 10  # puedes ajustar el máximo de concurrencia
    ):
        context = context or {}
        exchanges = exchanges or ["kraken", "binance", "coinbase", "alpaca"]

        results_by_exchange = collections.defaultdict(list)
        summary = collections.defaultdict(lambda: {
            "symbols_analyzed": 0,
            "executed": 0,
            "actions": collections.Counter(),
            "strategies": collections.Counter(),
            "total_profit": 0.0,
            "executed_symbols": [],
        })

        semaphore = asyncio.Semaphore(concurrency)

        async def analyze_and_execute(symbol, exchange):
            async with semaphore:
                print(f"[PARALLEL] Analizando {symbol} en {exchange}...")
                try:
                    analysis_list = await self.analyze(symbol, {**context, "exchange": exchange})
                    if isinstance(analysis_list, dict):
                        analysis_list = [analysis_list]
                    for analysis in analysis_list:
                        analysis["symbol"] = symbol
                        analysis["exchange"] = exchange
                        results_by_exchange[exchange].append(analysis)
                        summary[exchange]["symbols_analyzed"] += 1
                        action = (analysis.get("decision") or "hold").lower()
                        strategy = analysis.get("selected_strategy") or analysis.get("strategy") or "N/A"
                        summary[exchange]["actions"][action] += 1
                        summary[exchange]["strategies"][strategy] += 1

                        profit = float(
                            (analysis.get("order_result", {}) or {}).get("pnl_usd")
                            or analysis.get("estimated_roi")
                            or 0
                        )
                        if action in ["buy", "long_buy", "scalping", "executed"]:
                            summary[exchange]["total_profit"] += profit
                            summary[exchange]["executed"] += 1
                            summary[exchange]["executed_symbols"].append(symbol)
                        elif action in ["sell", "short_sell"]:
                            summary[exchange]["total_profit"] -= abs(profit)
                            summary[exchange]["executed"] += 1
                            summary[exchange]["executed_symbols"].append(symbol)
                except Exception as e:
                    self.logger.error(f"[{symbol}/{exchange}] Error en análisis institucional: {e}")

        # Crear todas las tareas en paralelo
        tasks = [
            analyze_and_execute(symbol, exchange)
            for exchange in exchanges
            for symbol in symbols
        ]

        await asyncio.gather(*tasks)

        print("\n=== RESUMEN INSTITUCIONAL POR EXCHANGE ===")
        for exchange in exchanges:
            data = summary[exchange]
            print(f"\n📊 Exchange: {exchange.upper()}")
            print(f" - Símbolos analizados: {data['symbols_analyzed']}")
            print(f" - Ejecuciones: {data['executed']}")
            print(f" - Acciones ejecutadas: {dict(data['actions'])}")
            print(f" - Estrategias usadas: {dict(data['strategies'])}")
            print(f" - Profit neto: {data['total_profit']:.4f}")
            print(f" - Symbols ejecutados: {data['executed_symbols']}")

        return results_by_exchange, summary
    async def simulate_alternative_path(self, context, actual_decision, alternative_decision):
        print(f"🔄 Simulación alternativa: {actual_decision} vs {alternative_decision} en {context}")
        return {"outcome_difference": "+1.8%", "confidence": 0.78}

    async def analyze_reaction(df: pd.DataFrame, returns: Optional[list[float]] = None) -> dict:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning("⚠️ analyze_reaction recibió un objeto no DataFrame o vacío.")
                return {
                    "status": "error",
                    "reason": "No se recibió un DataFrame válido para análisis."
                }

            df = await apply_indicators(df)

            rsi = df["rsi"].iloc[-1]
            atr = df["atr"].iloc[-1]
            supertrend = df["supertrend"].iloc[-1]
            vwap = df["vwap"].iloc[-1]
            close = df["close"].iloc[-1]
            adx = df["adx"].iloc[-1]
            boll_upper = df["upper_band"].iloc[-1]
            boll_lower = df["lower_band"].iloc[-1]

            estrategia = "none"
            if adx > 25 and supertrend > 0 and rsi < 70 and close > vwap:
                estrategia = "sniper"
            elif rsi < 30:
                estrategia = "rebound"
            elif close < boll_lower:
                estrategia = "oversold_bounce"
            elif close > boll_upper:
                estrategia = "overbought_fade"
            elif atr < 0.01 * close:
                estrategia = "consolidation_breakout"
            else:
                estrategia = "scalp"

            # Métricas adicionales si returns está presente
            sharpe = calculate_sharpe_ratio(returns) if returns else None
            win_rate = calculate_win_rate(returns) if returns else None
            max_drawdown = calculate_max_drawdown(returns) if returns else None

            result = {
                "status": "ok",
                "strategy": estrategia,
                "metrics": {
                    "rsi": round(rsi, 2),
                    "adx": round(adx, 2),
                    "atr": round(atr, 4),
                    "supertrend": supertrend,
                    "vwap": round(vwap, 2),
                    "close": round(close, 2),
                    "boll_upper": round(boll_upper, 2),
                    "boll_lower": round(boll_lower, 2),
                    "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
                    "win_rate": round(win_rate * 100, 2) if win_rate is not None else None,
                    "max_drawdown": round(max_drawdown, 3) if max_drawdown is not None else None
                }
            }

            logger.info(f"🧠 analyze_reaction completado para activo. Estrategia: {estrategia}")
            return result

        except Exception as e:
            logger.error(f"❌ analyze_reaction falló: {e}", exc_info=True)
            return {
                "status": "error",
                "reason": str(e)
            }
    async def detect_behavioral_bias(self, user_id):
        await self.load_memory()
        logs = self.memory.get(user_id, [])
        impulsive_moves = [a for a in logs if a["mode"] == "phantom" and a["decision"] == "entry too early"]
        return "bias:impulsivity" if len(impulsive_moves) > 3 else "bias:none"

    async def generate_user_tactical_profile(self, user_id):
        await self.load_memory()
        logs = self.memory.get(user_id, [])
        entries = sum(1 for x in logs if "entry" in x["decision"])
        reentries = sum(1 for x in logs if "reentry" in x["decision"])
        aborts = sum(1 for x in logs if "abort" in x["decision"])
        if aborts > entries:
            return "profile:conservative"
        elif reentries > entries * 0.5:
            return "profile:aggressive"
        else:
            return "profile:balanced"

    async def plan_dynamic_rotation(self, portfolio: dict, context: dict, market_symbols: list = None,
                                    min_expected_profit: float = 0.5) -> list:
        """
        Analiza el portafolio y genera un plan de rotación de capital si se detecta una oportunidad superior.
        :param portfolio: dict con activos actuales y cantidades (ej: {"XRP": 500, "BTC": 0.5, ...})
        :param context: contexto del usuario/capital
        :param market_symbols: lista de símbolos a analizar (opcional, si no, toma todos los favoritos)
        :param min_expected_profit: profit (%) mínimo esperado para rotar
        :return: lista de acciones: [{"action": "sell", "symbol": "XRP", ...}, {"action": "buy", "symbol": "ADA", ...}]
        """
        actions = []
        market_symbols = market_symbols or FAVORITE_SYMBOLS
        user_capital = context.get("capital", 10000)
        # 1. Analiza cada símbolo del mercado y calcula el mejor expected ROI
        analyses = []
        for sym in market_symbols:
            ana_result = await self.analyze(sym, context)
            # Puede ser lista o dict, así que iteramos siempre como lista
            for ana in (ana_result if isinstance(ana_result, list) else [ana_result]):
                if ana.get("error"):
                    continue
                analyses.append(ana)
        # 2. Encuentra el mejor símbolo para rotar (mayor ROI esperado)
        best = max(analyses, key=lambda x: x.get("estimated_roi", 0), default=None)
        if not best or best.get("estimated_roi", 0) < min_expected_profit:
            return []  # No hay oportunidad clara
        # 3. Encuentra el activo en portafolio con menor expected ROI
        current_holdings = [a for a in analyses if portfolio.get(a["symbol"], 0) > 0]
        if not current_holdings:
            return []
        worst = min(current_holdings, key=lambda x: x.get("estimated_roi", 0))
        # 4. Si el “salto” de ROI justifica el cambio, propone la rotación
        jump = best["estimated_roi"] - worst["estimated_roi"]
        if jump < min_expected_profit:
            return []
        qty_to_sell = portfolio[worst["symbol"]] * 0.7  # Puedes ajustar el % a vender
        actions.append({
            "action": "sell",
            "symbol": worst["symbol"],
            "qty": qty_to_sell,
            "reason": f"Rotar a {best['symbol']} por mayor ROI"
        })
        # Supón que conviertes a USDT, luego compras el nuevo
        price_best = best.get("price", 0)
        price_worst = worst.get("price", 0)
        if price_best > 0 and price_worst > 0:
            qty_best = (qty_to_sell * price_worst) / price_best
            actions.append({
                "action": "buy",
                "symbol": best["symbol"],
                "qty": qty_best,
                "reason": f"Aprovechar momentum actual en {best['symbol']}"
            })
            # Añade Take Profit (TP), Stop Loss (SL) y reverse automático
            tp_percent = 0.03  # Take Profit al 3%
            sl_percent = 0.01  # Stop Loss al 1%
            actions.append({
                "action": "set_tp_sl",
                "symbol": best["symbol"],
                "tp": best["price"] * (1 + tp_percent),
                "sl": best["price"] * (1 - sl_percent),
                "reason": f"Take Profit en +{int(tp_percent * 100)}%, Stop Loss en -{int(sl_percent * 100)}%"
            })
            actions.append({
                "action": "monitor_and_reverse",
                "from": best["symbol"],
                "to": worst["symbol"],
                "tp": tp_percent,
                "sl": sl_percent,
                "reason": f"Revertir a {worst['symbol']} tras TP/SL en {best['symbol']}"
            })
        return actions
    async def analyze_performance(self, user_id, filter_by="all"):
        await self.load_memory()
        actions = self.memory.get(user_id, [])
        if filter_by == "tournament":
            filtered = [x for x in actions if x.get("context", {}).get("type") == "tournament"]
        elif filter_by == "phantom":
            filtered = [x for x in actions if x.get("mode") == "phantom"]
        elif filter_by == "real":
            filtered = [x for x in actions if x.get("context", {}).get("type") == "live"]
        else:
            filtered = actions
        win_count = sum(1 for x in filtered if x.get("result") == "win")
        total = len(filtered)
        avg_psi = sum(x.get("context", {}).get("psi", 0) for x in filtered) / total if total > 0 else 0
        return {
            "type": filter_by,
            "total_actions": total,
            "wins": win_count,
            "win_rate": round(win_count / total, 3) if total > 0 else 0,
            "avg_psi": round(avg_psi, 2)
        }

    async def find_profitable_symbols(self, context: dict, top_n: int = 3) -> List[str]:
        try:
            logger.info("🔍 Buscando símbolos con mayor potencial de profit...")
            all_symbols = await self.exchange.get_all_symbols()
            results = []

            for symbol in all_symbols:
                try:
                    analysis = await self.analyze(symbol, context)
                    if analysis.get("error"):
                        continue

                    roi = analysis.get("estimated_roi", 0)
                    manipulation = analysis.get("anomaly", {}).get("manipulation_risk_score", 0)
                    signal = analysis.get("signal_strength", 0)

                    # Filtro de calidad
                    if roi > 0.3 and manipulation < 0.5 and signal > 0.5:
                        results.append((symbol, roi, signal))
                except Exception as e:
                    logger.warning(f"❌ Error evaluando {symbol}: {e}")

            # Ordenar por ROI descendente y devolver los top_n símbolos
            results.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in results[:top_n]]
        except Exception as e:
            logger.error(f"❌ Error buscando símbolos rentables: {e}")
            return []

    async def adjust_reinforcement_policy(self, user_id, reward):
        print(f"⚙️ Ajustando política RL de {user_id} con reward: {reward}")

    async def generate_counter_strategy(self, opponent_style):
        if opponent_style == "momentum":
            return "stealth_delay + spoof_echo"
        elif opponent_style == "scalper":
            return "liquidity_trap + partial_fade"
        return "fragmented_entry + bait"

    async def apply_gamma_scalping(self, symbol: str, df: pd.DataFrame, user_id: int, capital: float) -> dict:
        """
        Ejecuta gamma_scalping si las condiciones lo permiten y devuelve la acción tomada.
        """
        self.logger.info(f"[{symbol}] 🧪 Evaluando gamma_scalping...")

        try:
            # Instancia flexible (por si cambia la firma del __init__)
            scalper = None
            try:
                scalper = GammaScalper(user_id=user_id, logger=self.logger, capital=capital, symbol=symbol)
            except TypeError:
                try:
                    scalper = GammaScalper(user_id=user_id)
                except TypeError:
                    scalper = GammaScalper()

            # Llama al método disponible (run o execute)
            if hasattr(scalper, "run"):
                out = await scalper.run(symbol=symbol, df=df, capital=capital, user_id=user_id, logger=self.logger)
            elif hasattr(scalper, "execute"):
                out = await scalper.execute(symbol=symbol, df=df, capital=capital, user_id=user_id, logger=self.logger)
            else:
                raise AttributeError("GammaScalper no expone run/execute.")

            # Normaliza salida (acepta tuple o dict)
            if isinstance(out, tuple) and len(out) >= 2:
                result, pnl = out[0], out[1]
            elif isinstance(out, dict):
                result = out.get("action") or out.get("result") or out.get("status")
                pnl = out.get("pnl", 0.0)
            else:
                result, pnl = None, 0.0

            if result in ("scalp", "executed", "buy", "sell"):
                self.logger.info(f"[{symbol}] ✅ Gamma Scalping ejecutado. PnL: {pnl:.4f}")
                return {"status": "executed", "strategy": "gamma_scalping", "pnl": float(pnl)}
            else:
                self.logger.info(f"[{symbol}] 🚫 Gamma Scalping no ejecutado.")
                return {"status": "skipped", "reason": "no_trade"}

        except Exception as e:
            self.logger.error(f"[{symbol}] ❌ Error ejecutando gamma_scalping: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    async def train_drl_for_symbol(self, symbol: str, episodes: int = 100):
        df = await fetch_crypto_data(symbol)
        if df is None or df.empty:
            print(f"[⚠️] No data for {symbol}")
            return
        env = TradingEnv(df)
        agent = PPOAgent(env=env)
        await agent.train(episodes=episodes)

    async def run_strategy(self, strategy_name: str, symbol: str, context: dict) -> dict:
        try:
            module_path = f'krypnova.strategies.{strategy_name}'
            strategy_module = importlib.import_module(module_path)

            if hasattr(strategy_module, 'run'):
                result = await strategy_module.run(symbol, context)
                return {
                    "strategy": strategy_name,
                    "result": result,
                    "status": "success"
                }
            else:
                return {
                    "strategy": strategy_name,
                    "status": "no_run_function"
                }

        except ModuleNotFoundError:
            return {
                "strategy": strategy_name,
                "status": "not_found"
            }
        except Exception as e:
            return {
                "strategy": strategy_name,
                "status": "error",
                "error": str(e)
            }

    STRATEGY_EXECUTORS = {
        "sniper": SniperMode,
        "bolivar_mode": BolivarMode,
        "shadow": ShadowStrategy,
        "hawk": HawkMode,
        "eclipse_protocol": EclipseProtocol,
        "phantom": PhantomTactics,
        "phoenix_protocol": PhoenixProtocol,
        "counter": CounterTacticEngine,
        "hydra": simple_hydra_launcher,
        "memecoin_sniper": MemecoinSniper,
        "gap_closer": run_gap_strategy,
        "htf_breakout": HTFBreakoutEntry,
        "news_reversal": NewsReversalEntry,
        "liquidity_sweeper": LiquiditySweeper,
        "trap_reversal": TrapReversalEngine,
        "stop_hunt": StopHuntExecutor,
        "momentum": MomentumDriver,
        "rebound": ReboundEntry,
        "arbitrage": ArbitrageSystem,
        "evasion": EvasionManager,
        "sentinel": SentinelController,
        "gamma_scalper": GammaScalper
    }

    async def make_decision(
            self,
            symbol: str | list,
            context: dict,
            all_analyses: list = None,
            auto_switch: bool = True,
            tested_symbols: set = None,
            batch_mode: bool = False,
            top_n: int = 3,
            portfolio: dict = None,
            allow_rotation: bool = True,
            min_rotation_profit: float = 0.5
    ) -> dict:
        """
        Toma una decisión táctica para uno o varios símbolos dados.
        Si recibe una lista, procesa cada símbolo individualmente y en paralelo.
        """
        logger = getattr(self, "logger", logging.getLogger("ExionBrain"))
        try:
            # Si symbol es lista/iterable (pero NO string), procesa todos en paralelo
            if isinstance(symbol, (list, tuple, set)) and not isinstance(symbol, str):
                tasks = [
                    self.make_decision(
                        sym,
                        context,
                        all_analyses=all_analyses,
                        auto_switch=auto_switch,
                        tested_symbols=tested_symbols,
                        batch_mode=batch_mode,
                        top_n=top_n,
                        portfolio=portfolio,
                        allow_rotation=allow_rotation,
                        min_rotation_profit=min_rotation_profit
                    )
                    for sym in symbol
                ]
                results = await asyncio.gather(*tasks)
                return {sym: res for sym, res in zip(symbol, results)}

            # --- Proceso normal para UN símbolo ---
            logger.info(f"🧠 Tomando decisión táctica para {symbol}...")

            tested_symbols = tested_symbols or set()
            tested_symbols.add(symbol)

            offensive_selector_names = context.get("offensive_selector", [])
            if isinstance(offensive_selector_names, str):
                offensive_selector_names = [offensive_selector_names]
            offensive_selector_names = [s.lower() for s in offensive_selector_names]

            # Ejecutar estrategias ofensivas si existen
            offensive_results = []
            for strat_name in offensive_selector_names:
                result = await self.offensive_selector.run_strategies_for_symbols(strat_name, symbol, context)
                if isinstance(result, list):
                    offensive_results.extend(result)
                else:
                    offensive_results.append(result)

            # Elegir la mejor estrategia ofensiva (mayor ROI)
            best_offensive_result = None
            best_offensive_roi = float("-inf")
            for res in offensive_results:
                if isinstance(res, dict):
                    roi = res.get("roi", 0.0)
                    try:
                        roi = float(roi)
                    except Exception:
                        roi = 0.0
                    if roi > best_offensive_roi:
                        best_offensive_roi = roi
                        best_offensive_result = res

            # --- Ejecutar análisis principal ---
            analyses = await self.analyze(symbol, context)
            flat_analyses = []
            if isinstance(analyses, dict):
                flat_analyses = [analyses]
            elif isinstance(analyses, list):
                for ana in analyses:
                    if isinstance(ana, list):
                        flat_analyses.extend(ana)
                    else:
                        flat_analyses.append(ana)
            else:
                logger.error(f"analyze retornó tipo inesperado: {type(analyses)} - {analyses}")
                flat_analyses = []

            analyses = flat_analyses

            # --- Selección robusta del mejor análisis con defensa priorizada ---
            valid_analysis = None

            for analysis in analyses:
                if not isinstance(analysis, dict):
                    logger.error(f"analysis retornó tipo inesperado: {type(analysis)} - {analysis}")
                    continue
                decision = analysis.get("decision", "").lower()
                defense = analysis.get("defense", {})
                # Nueva lógica: prioriza defensa si bloquea
                defense_decision = (defense.get("final_decision") or defense.get("decision") or "").upper()
                if defense_decision in ["BLOCK", "WAIT", "HOLD"]:
                    analysis["decision"] = "hold"
                    analysis["reason"] = f"Defensa bloquea/espera: {defense.get('reason', 'Riesgo detectado')}"
                    analysis["strategy_used"] = "defense"
                    valid_analysis = analysis
                    break
                if decision in ["buy", "sell", "long", "short"]:
                    # Si defensa permite (ACT/CLEAR)
                    if defense_decision in ["ACT", "CLEAR", ""]:
                        valid_analysis = analysis
                        break

            # --- Si ninguna defensa bloquea ni hay señal ofensiva clara, fuerza ofensiva si hay buen ROI/sharpe/etc ---
            if not valid_analysis and best_offensive_result:
                valid_analysis = best_offensive_result
                valid_analysis["defense_overridden"] = True
                valid_analysis["reason"] = "Ejecución forzada por mejor estrategia ofensiva (ignora defensa)."
            elif not valid_analysis:
                offensive_candidates = []
                for analysis in analyses:
                    if not isinstance(analysis, dict):
                        continue
                    decision = analysis.get("decision") or ""
                    defense = analysis.get("defense", {})
                    defense_decision = (defense.get("final_decision") or defense.get("decision") or "").upper()
                    norm_decision = (decision or "").replace("_", " ").strip().lower()
                    # Permitir sólo si defensa permite o está vacío
                    if (defense_decision in ["ACT", "CLEAR", ""] and norm_decision in offensive_selector_names) or (
                        not offensive_selector_names and is_executable_action(decision)):
                        roi = analysis.get("signals", {}).get("signal_strength", 0) or analysis.get("portfolio_metrics", {}).get("expected_return", 0)
                        sharpe = analysis.get("portfolio_metrics", {}).get("sharpe_ratio", 0)
                        pattern_score = analysis.get("patterns_score", {}).get("score", 0)
                        sentiment = analysis.get("signals", {}).get("sentiment_score", 0)
                        is_positive = (
                            roi > 0.10 or
                            sharpe > 0.7 or
                            pattern_score > 0.5 or
                            sentiment > 0.2
                        )
                        if is_positive:
                            analysis["offensive_forced"] = True
                            offensive_candidates.append((roi, analysis))
                if offensive_candidates:
                    offensive_candidates.sort(reverse=True, key=lambda x: x[0])
                    valid_analysis = offensive_candidates[0][1]
                    valid_analysis["defense_overridden"] = True
                    valid_analysis["reason"] = "Ejecución forzada por señal ofensiva positiva (ignora defensa)."

            # --- Si no hay oportunidad, evalúa rotación o HOLD ---
            if not valid_analysis:
                if allow_rotation and portfolio:
                    for sym in portfolio:
                        if not isinstance(portfolio[sym], dict):
                            price = None
                            try:
                                price_analysis = await self.analyze(sym, context)
                                pa = price_analysis[0] if isinstance(price_analysis, list) and price_analysis else price_analysis
                                if pa and isinstance(pa, dict):
                                    price = pa.get("price") or pa.get("entry_price")
                            except Exception:
                                price = None
                            portfolio[sym] = {"qty": float(portfolio[sym]), "price": float(price) if price else 0.0}
                    rotation_plan = await self.plan_dynamic_rotation(
                        portfolio=portfolio,
                        context=context,
                        min_expected_profit=min_rotation_profit
                    )
                    if rotation_plan:
                        rotation_result = {
                            "symbol": symbol,
                            "decision": "rotate_portfolio",
                            "reason": "Se recomienda rotar activos para maximizar rentabilidad",
                            "rotation_plan": rotation_plan,
                            "confidence": 0.7,
                            "strategy_mode": "rotation"
                        }
                        user_id = context.get("user_id", "unknown")
                        await save_decision_db(self, user_id, symbol, rotation_result.get("decision"), context, rotation_result)
                        await self.record_action(
                            user_id=user_id,
                            mode="make_decision",
                            context=context,
                            decision=rotation_result.get("decision"),
                            result=rotation_result,
                        )
                        return rotation_result
                hold_result = {
                    "symbol": symbol,
                    "decision": "hold",
                    "reason": "No se encontraron oportunidades claras o la defensa bloquea/espera la operación.",
                    "strategy_used": "N/A"
                }
                user_id = context.get("user_id", "unknown")
                await save_decision_db(self, user_id, symbol, hold_result.get("decision"), context, hold_result)
                await self.record_action(
                    user_id=user_id,
                    mode="make_decision",
                    context=context,
                    decision=hold_result.get("decision"),
                    result=hold_result,
                )
                return hold_result

            # --- Calcular ROI total ---
            roi_total = None
            if "roi" in valid_analysis:
                roi_total = valid_analysis.get("roi")
            elif "signals" in valid_analysis and "signal_strength" in valid_analysis["signals"]:
                roi_total = valid_analysis["signals"]["signal_strength"]
            elif "portfolio_metrics" in valid_analysis and "expected_return" in valid_analysis["portfolio_metrics"]:
                roi_total = valid_analysis["portfolio_metrics"]["expected_return"]
            else:
                roi_total = 0.0

            try:
                if roi_total is None or not isinstance(roi_total, (int, float)):
                    roi_total = 0.0
            except Exception:
                roi_total = 0.0

            valid_analysis["roi_total"] = roi_total

            profile = context.get("profile", getattr(self, "profile", "moderate"))

            def should_auto_execute(analysis: dict, profile: str) -> bool:
                def _to_01(x):
                    if x is None:
                        return None
                    try:
                        x = float(x)
                        return x / 100.0 if x > 1.0 else x
                    except Exception:
                        return None

                mc_prob = _to_01((analysis.get("montecarlo") or {}).get("probabilidad_ganancia"))
                bt_prob = _to_01((analysis.get("metrics") or {}).get("win_rate"))
                q_conf = analysis.get("quick_decision", {}).get("confidence")
                q_prob = _to_01(q_conf)

                candidates = [p for p in (mc_prob, bt_prob, q_prob) if isinstance(p, float)]
                combined_prob = sum(candidates) / len(candidates) if candidates else 0.0

                logger.info(f"[AutoExec] Probabilidad combinada: {combined_prob:.2%} | Perfil: {profile}")

                if profile == "aggressive":
                    return combined_prob >= 0.45
                if profile == "moderate":
                    return combined_prob >= 0.55
                if profile == "conservative":
                    return combined_prob >= 0.60
                return combined_prob >= 0.55

            if should_auto_execute(valid_analysis, profile):
                valid_analysis["execution_mode"] = "auto_execute"
                logger.info(
                    f"[AutoExec] Ejecutando {valid_analysis['decision']} en {valid_analysis.get('symbol', symbol)} por superar umbral."
                )
                try:
                    side = valid_analysis["decision"]
                    if hasattr(self, "executor") and hasattr(self.executor, "execute_order"):
                        order_result = await self.executor.execute_order(
                            symbol=valid_analysis.get("symbol", symbol),
                            side=side,
                            qty=valid_analysis.get("final_position_size", 0.01),
                            price=valid_analysis.get("price")
                        )
                        valid_analysis["execution_result"] = order_result
                except Exception as e:
                    logger.error(f"❌ Error ejecutando orden para {valid_analysis.get('symbol', symbol)}: {e}")
                    valid_analysis["execution_result"] = {"error": str(e)}

            user_id = context.get("user_id", "unknown")
            await save_decision_db(self, user_id, symbol, valid_analysis.get("decision"), context, valid_analysis)
            await self.record_action(
                user_id=user_id,
                mode="make_decision",
                context=context,
                decision=valid_analysis.get("decision"),
                result=valid_analysis,
            )
            return valid_analysis

        except Exception as e:
            logger.error(f"❌ Error tomando decisión para {symbol}: {e}", exc_info=True)
            user_id = context.get("user_id", "unknown")
            error_result = {"symbol": symbol, "error": str(e)}
            await save_decision_db(self, user_id, symbol, "error", context, error_result)
            await self.record_action(
                user_id=user_id,
                mode="make_decision",
                context=context,
                decision="error",
                result=error_result,
            )
            return error_result
    async def run_full_exion_brain_scan(
            context: dict,
            top_n: int = 5,
            verbose: bool = True,
            **kwargs
    ):
        favoritos = set(context.get("favoritos", []))
        mover_fetcher = TopMoversFetcher()
        all_movers = await mover_fetcher.get_all_movers(limit=10)
        movers_symbols = set()
        for movers in all_movers:
            movers_symbols.update([s for s, _ in movers.get("gainers", [])])
            movers_symbols.update([s for s, _ in movers.get("losers", [])])
        symbols_to_analyze = list(favoritos) + [s for s in movers_symbols if s not in favoritos]

        results = []
        for symbol in symbols_to_analyze:
            try:
                outcome, metrics = await execute_exion_brain(symbol, context, **kwargs)
                # Filtra por ROI y Sharpe positivos
                roi = outcome.get("roi")
                sharpe = metrics.get("sharpe")
                if (
                        outcome.get("decision") in ["buy", "sell"] and
                        roi is not None and roi > 0 and
                        sharpe is not None and sharpe > 0
                ):
                    outcome["roi_sharpe_score"] = roi * sharpe
                    results.append(outcome)
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")

        # Ordena y selecciona top N
        top_results = sorted(
            results,
            key=lambda x: x.get("roi_sharpe_score", 0),
            reverse=True
        )[:top_n]

        # Opcional: lanzar alertas/acciones
        for opp in top_results:
            await send_alert(opp)
            # O ejecutar trade: await execute_trade(opp)

        if verbose:
            print("\n=== TOP OPORTUNIDADES DEL DÍA ===")
            for opp in top_results:
                print(
                    f"SYMBOL: {opp['symbol']} | ROI: {opp['roi']:.3f} | SHARPE: {opp['full_decision'].get('sharpe'):.2f}")

        return top_results

    async def run_exion_brain_daily_top5(
            context: dict,
            top_n: int = 5,
            profile: str = "moderate",
            capital: float = 10000,
            user_id: str = "default_user",
            run_backtest: bool = True,
            run_montecarlo: bool = True,
            save_memory: bool = True,
            verbose: bool = True,
            **kwargs,
    ):
        """
        Orquesta todo el pipeline de ExionBrain para favoritos + top movers,
        selecciona los top N del día por ROI x Sharpe y ejecuta alertas/acciones.
        """
        favoritos = set(context.get("favoritos", []))
        results = []

        # 1. Obtener top movers del día
        mover_fetcher = TopMoversFetcher()
        all_movers = await mover_fetcher.get_all_movers(limit=10)
        movers_symbols = set()
        for movers in all_movers:
            movers_symbols.update([s for s, _ in movers.get("gainers", [])])
            movers_symbols.update([s for s, _ in movers.get("losers", [])])

        # 2. Lista de símbolos a analizar: favoritos + movers no favoritos
        symbols_to_analyze = list(favoritos) + [s for s in movers_symbols if s not in favoritos]

        # 3. Analizar cada símbolo con ExionBrain
        for symbol in symbols_to_analyze:
            try:
                outcome, metrics = await execute_exion_brain(
                    symbol=symbol,
                    context=context,
                    profile=profile,
                    capital=capital,
                    user_id=user_id,
                    run_backtest=run_backtest,
                    run_montecarlo=run_montecarlo,
                    save_memory=save_memory,
                    verbose=False,  # solo muestra resumen final
                    **kwargs
                )
                roi = outcome.get("roi")
                sharpe = metrics.get("sharpe")
                if (
                        outcome.get("decision") in ["buy", "sell"] and
                        roi is not None and roi > 0 and
                        sharpe is not None and sharpe > 0
                ):
                    outcome["roi_sharpe_score"] = roi * sharpe
                    results.append(outcome)
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")

        # 4. Ordenar y elegir top N
        top_results = sorted(
            results,
            key=lambda x: x.get("roi_sharpe_score", 0),
            reverse=True
        )[:top_n]

        # 5. Alertar o ejecutar acción sobre cada uno de los top N
        for opp in top_results:
            await send_alert(opp)  # Aquí tu función de alerta
            # También puedes ejecutar trade: await execute_trade(opp)

        if verbose:
            print("\n=== TOP OPORTUNIDADES DEL DÍA ===")
            for opp in top_results:
                print(
                    f"SYMBOL: {opp['symbol']} | ROI: {opp['roi']:.3f} | SHARPE: {opp['full_decision'].get('sharpe'):.2f}")

        return top_results
FAVORITE_SYMBOLS = ["XRPUSD", "BTCUSD", "ADAUSD", "ETHUSD", "SOLUSD", "BNBUSD"]


def random_symbol(exclude=None):
    syms = [s for s in FAVORITE_SYMBOLS if s != exclude]
    return random.choice(syms)



async def execute_exion_brain(
    symbol: str,
    context: dict = None,
    profile: str = "moderate",
    capital: float = 10000,
    user_id: str = "default_user",
    candidate_assets: list = None,
    run_backtest: bool = True,
    run_montecarlo: bool = True,
    save_memory: bool = True,
    verbose: bool = True,
    **kwargs
):
    exion = ExionBrain(user_id=user_id, profile=profile, capital=capital)
    if context is None:
        context = {}
    context = {**context, "user_id": user_id, "profile": profile, "capital": capital}
    if candidate_assets:
        context["assets"] = candidate_assets

    decision = await exion.make_decision(symbol, context)

    if save_memory:
        await exion.record_action(
            user_id=user_id,
            mode="exion_brain",
            context=context,
            decision=decision.get("decision"),
            result=decision
        )

    if verbose:
        print(f"\n=== EXION BRAIN RUN for {symbol} ===")
        print("DECISION:", decision.get("decision"))
        print("STRATEGY:", decision.get("strategy_used", decision.get('strategy_mode', 'N/A')))
        print("ROI:", decision.get("estimated_roi"))
        print("MONTECARLO:", decision.get("montecarlo"))
        print("BACKTEST:", decision.get("backtest"))
        print("===============================")

    # Bloque de métricas para debug
    metrics = {
        "win_rate": decision.get("win_rate"),
        "sharpe": decision.get("sharpe"),
        "roi": decision.get("estimated_roi"),
        "confidence": decision.get("confidence"),
        "mc_prob": decision.get("montecarlo_prob"),
        "defense_status": decision.get("defense_status"),
    }
    return {
        "symbol": symbol,
        "decision": decision.get("decision"),
        "strategy": decision.get("strategy_used", decision.get('strategy_mode', 'N/A')),
        "roi": decision.get("estimated_roi"),
        "montecarlo": decision.get("montecarlo"),
        "backtest": decision.get("backtest"),
        "full_decision": decision
    }, metrics

async def analyze_and_execute(exion, symbol, context, portfolio, tested_symbols, run_id):
    """
    Analiza el símbolo y ejecuta la acción si es pertinente.
    Retorna el resultado de la acción.
    """
    tested_symbols.add(symbol)
    decision = await exion.make_decision(symbol, context)
    strategy_used = decision.get("strategy_used", decision.get("strategy_mode", "N/A"))
    action = decision.get("decision", "").lower()

    print(f"\n[RUN_ID={run_id}] --- TEST: {symbol} ---")
    print(f"[RUN_ID={run_id}] ANALYSIS DECISION:", decision)
    print(f"[RUN_ID={run_id}] 🔎 Estrategia usada:", strategy_used)
    print(f"[RUN_ID={run_id}] ➡️ Decisión tomada:", action)
    print(f"[RUN_ID={run_id}] 💰 Estimación ROI:", decision.get("estimated_roi"))
    print(f"[RUN_ID={run_id}] 🎲 MonteCarlo:", decision.get("montecarlo"))
    print(f"[RUN_ID={run_id}] 📈 Backtest:", decision.get("backtest_metrics"))

    # Normalización de acciones
    profit = 0
    if action in ["long_buy", "buy"] and decision.get("estimated_roi", 0) > 0:
        mc_prob = decision.get("montecarlo", {}).get("probabilidad_ganancia", 0) or 0.5
        profit = decision.get("estimated_roi", 1) * mc_prob
        portfolio[symbol] += profit
    elif action in ["short_sell", "sell"]:
        mc_prob = decision.get("montecarlo", {}).get("probabilidad_ganancia", 0) or 0.5
        profit = abs(decision.get("estimated_roi", 1)) * mc_prob
        portfolio[symbol] += profit

    return {
        "symbol": symbol,
        "decision": decision,
        "profit": profit
    }

async def run_once():
    exion = ExionBrain(user_id="demo_user", profile="moderate", capital=10_000)
    symbols = ["ETH-USD", "BTC-USD", "XBTUSD", "ETHUSD", "SPY", "NVDA", "AAPL"]

    ctx_base = {
        "profile": "moderate",
        "capital": 10_000,
        "user_id": exion.user_id,
        "connector": ExchangeConnector(),
        "clones_manager": ClonesManager(),
        "liquidity": 500000,
        "volume_24h": 1000000,
        "volatility": 0.05,
        "user_capital": 10_000,
        "session": "mysession",
        "email": "user@email.com",
    }
    exchanges = ["kraken", "coinbase", "alpaca"]

    for symbol in symbols:
        for exchange in exchanges:
            normalized_symbol = exion._normalize_symbol(symbol, exchange)
            print(f"Consultando {exchange.upper()} con símbolo: {normalized_symbol}")

        for symbol in symbols:
            ctx = dict(ctx_base)
            ctx["asset"] = symbol
            ctx["assets"] = [symbol]
            ctx.setdefault("clones_manager", ClonesManager())
            ctx.setdefault("connector", ExchangeConnector())
            ctx.setdefault("liquidity", 500000)
            ctx.setdefault("volume_24h", 1000000)
            ctx.setdefault("volatility", 0.05)
            ctx.setdefault("user_capital", 10000)
            ctx.setdefault("ticker", symbol)
            ctx.setdefault("session", "mysession")
            ctx.setdefault("email", "demo@email.com")
            ctx["strategies"] = make_strategies(ctx, symbol)
            # ...
        results = await exion.make_decision(symbol=symbol, context=ctx)
        print(f"\n====== RESULTS for {symbol} ======")
        print(results)
        print("==================================\n")

def print_exion_decision(result):
    # Busca en el dict raíz
    strat = result.get('strategy')
    roi = result.get('roi_total', result.get('roi'))

    # Si no encuentra, busca en 'full_decision' o en cualquier subdict
    if strat is None or roi is None:
        if 'full_decision' in result and isinstance(result['full_decision'], dict):
            fd = result['full_decision']
            if strat is None:
                strat = fd.get('strategy')
            if roi is None:
                roi = fd.get('roi_total', fd.get('roi'))
        else:
            # Busca en cualquier subdict por si acaso
            for v in result.values():
                if isinstance(v, dict):
                    if strat is None:
                        strat = v.get('strategy')
                    if roi is None:
                        roi = v.get('roi_total', v.get('roi'))
                    if strat and roi:
                        break

    print(f"STRATEGY: {strat if strat is not None else 'N/A'}")
    print(f"ROI: {roi if roi is not None else 'None'}")

# Uso en tu main async:
if __name__ == "__main__":
    async def main():
        result, metrics = await execute_exion_brain(
            symbol=["BTC-USD", "ETH-USD", "NVDA", "PEPE-USD"],
            context={"profile": "moderate"},
            user_id="demo_user"
        )
        print("Métricas usadas para decidir:", {
            "win_rate": metrics.get("win_rate"),
            "sharpe": metrics.get("sharpe"),
            "roi": metrics.get("roi"),
            "confidence": metrics.get("confidence"),
            "mc_prob": metrics.get("mc_prob"),
            "defense_status": metrics.get("defense_status"),
        })
        print(result)
        print_exion_decision(result)

    import asyncio
    asyncio.run(main())