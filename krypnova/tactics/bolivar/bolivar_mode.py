"""
BolivarMode: Advanced trading strategy that leverages DecisionMapper
for standardized, profile-based decision making with comprehensive analytics.
"""
import logging
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

# Import the DecisionMapper from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_mapper import DecisionMapper

logger = logging.getLogger(__name__)


class BolivarMode:
    """
    Advanced BolivarMode strategy that integrates with DecisionMapper
    for robust, standardized decision making.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.decision_mapper = DecisionMapper()
        self.name = "BolivarMode"
        
        # Strategy configuration
        self.config = {
            "min_confidence": 0.6,
            "max_position_size": 0.1,  # 10% of capital
            "stop_loss_pct": 0.05,     # 5% stop loss
            "take_profit_pct": 0.15    # 15% take profit
        }
    
    def calculate_metrics(self, symbol: str, df: pd.DataFrame, capital: float) -> tuple:
        """
        Calculate comprehensive metrics for the symbol using available data.
        Returns: (roi, mc_roi, metrics, mc_results)
        """
        try:
            self.logger.info(f"Calculating metrics for {symbol} with capital {capital}")
            
            if df is None or df.empty:
                self.logger.warning(f"Empty dataframe for {symbol}")
                return 0.0, 0.0, {}, {}
            
            # Get the latest price data
            latest_price = self._get_latest_price(df)
            if latest_price <= 0:
                self.logger.warning(f"Invalid latest price for {symbol}: {latest_price}")
                return 0.0, 0.0, {}, {}
            
            # Calculate technical indicators
            technical_metrics = self._calculate_technical_indicators(df)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(df, capital)
            
            # Calculate volume and liquidity metrics
            volume_metrics = self._calculate_volume_metrics(df)
            
            # Estimate sentiment (simplified for now)
            sentiment_metrics = self._calculate_sentiment_metrics(df)
            
            # Run Monte Carlo simulation
            mc_results = self._run_monte_carlo_simulation(df, capital)
            
            # Combine all metrics
            combined_metrics = {
                **technical_metrics,
                **risk_metrics,
                **volume_metrics,
                **sentiment_metrics,
                "symbol": symbol,
                "latest_price": latest_price,
                "capital": capital,
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate estimated ROI
            roi = self._estimate_roi(combined_metrics)
            mc_roi = mc_results.get("roi_promedio", 0.0)
            
            self.logger.info(f"Metrics calculated - ROI: {roi:.4f}, MC ROI: {mc_roi:.4f}")
            
            return roi, mc_roi, combined_metrics, mc_results
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {symbol}: {e}", exc_info=True)
            return 0.0, 0.0, {}, {}
    
    def get_action_and_confidence(self, df: pd.DataFrame, metrics: Dict[str, Any], 
                                 mc_results: Dict[str, Any]) -> tuple:
        """
        Legacy method for compatibility. Now delegates to DecisionMapper.
        Returns: (action, confidence)
        """
        try:
            # Use default moderate profile if not specified
            profile = "moderate"
            
            # Prepare metrics for DecisionMapper
            decision_metrics = {**metrics, "montecarlo": mc_results}
            
            # Get decision from DecisionMapper
            decision = self.decision_mapper.map_decision(profile, decision_metrics)
            
            return decision.action, decision.confidence
            
        except Exception as e:
            self.logger.error(f"Error in get_action_and_confidence: {e}", exc_info=True)
            return "hold", 0.0
    
    async def run(self, symbol: str, df: pd.DataFrame, capital: float, 
                 user_profile: str = "moderate", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main run method that executes the BolivarMode strategy using DecisionMapper.
        
        Args:
            symbol: Trading symbol
            df: Price/volume dataframe
            capital: Available capital
            user_profile: Risk profile (conservative, moderate, aggressive)
            context: Additional context information
            
        Returns:
            Standardized decision result from DecisionMapper
        """
        try:
            self.logger.info(f"Running BolivarMode for {symbol} with profile {user_profile}")
            
            # Validate inputs
            if df is None or df.empty:
                return self._create_error_result(symbol, "Empty or invalid dataframe")
            
            if capital <= 0:
                return self._create_error_result(symbol, "Invalid capital amount")
            
            # Calculate all metrics
            roi, mc_roi, metrics, mc_results = self.calculate_metrics(symbol, df, capital)
            
            # Add context information if provided
            if context:
                metrics.update({
                    "user_id": context.get("user_id"),
                    "session": context.get("session"),
                    "exchange": context.get("exchange"),
                    "account_data": context.get("account_data", {})
                })
            
            # Prepare comprehensive metrics for DecisionMapper
            decision_metrics = {
                **metrics,
                "montecarlo": mc_results,
                "strategy_name": self.name,
                "roi_estimate": roi,
                "mc_roi_estimate": mc_roi
            }
            
            # Get decision from DecisionMapper
            decision_result = self.decision_mapper.map_decision(user_profile, decision_metrics)
            
            # Convert DecisionResult to dictionary format expected by the system
            result = self._format_decision_result(decision_result, symbol, roi, mc_roi, metrics, mc_results)
            
            # Add strategy-specific enhancements
            result = self._enhance_result_with_strategy_data(result, df, capital, context)
            
            self.logger.info(f"BolivarMode decision: {decision_result.action} with {decision_result.confidence:.2f} confidence")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in BolivarMode.run for {symbol}: {e}", exc_info=True)
            return self._create_error_result(symbol, str(e))
    
    def _get_latest_price(self, df: pd.DataFrame) -> float:
        """Get the latest closing price from the dataframe"""
        try:
            if 'close' in df.columns:
                return float(df['close'].iloc[-1])
            elif 'Close' in df.columns:
                return float(df['Close'].iloc[-1])
            elif len(df.columns) > 0:
                return float(df.iloc[-1, -1])  # Last column, last row
            return 0.0
        except (IndexError, ValueError, TypeError):
            return 0.0
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical analysis indicators"""
        try:
            metrics = {}
            
            # Ensure we have the required columns
            if 'close' in df.columns:
                close = df['close']
            elif 'Close' in df.columns:
                close = df['Close']
            else:
                close = df.iloc[:, -1]  # Assume last column is close
            
            # RSI calculation (simplified)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            metrics["rsi"] = float((100 - (100 / (1 + rs))).iloc[-1]) if len(rs) > 0 else 50.0
            
            # Simple Moving Average
            if len(close) >= 20:
                sma_20 = close.rolling(window=20).mean()
                metrics["sma_trend"] = float((close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1])
            else:
                metrics["sma_trend"] = 0.0
            
            # MACD (simplified)
            if len(close) >= 26:
                ema_12 = close.ewm(span=12).mean()
                ema_26 = close.ewm(span=26).mean()
                macd = ema_12 - ema_26
                metrics["macd"] = float(macd.iloc[-1])
            else:
                metrics["macd"] = 0.0
            
            # Bollinger Bands position
            if len(close) >= 20:
                sma = close.rolling(window=20).mean()
                std = close.rolling(window=20).std()
                upper_band = sma + (std * 2)
                lower_band = sma - (std * 2)
                bb_position = (close.iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
                metrics["bollinger_position"] = float(bb_position) if not pd.isna(bb_position) else 0.5
            else:
                metrics["bollinger_position"] = 0.5
            
            # EMA trend
            if len(close) >= 12:
                ema = close.ewm(span=12).mean()
                metrics["ema_trend"] = float((close.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1])
            else:
                metrics["ema_trend"] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {"rsi": 50.0, "macd": 0.0, "bollinger_position": 0.5, "sma_trend": 0.0, "ema_trend": 0.0}
    
    def _calculate_risk_metrics(self, df: pd.DataFrame, capital: float) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        try:
            metrics = {}
            
            # Get price column
            if 'close' in df.columns:
                prices = df['close']
            elif 'Close' in df.columns:
                prices = df['Close']
            else:
                prices = df.iloc[:, -1]
            
            # Volatility (standard deviation of returns)
            returns = prices.pct_change().dropna()
            if len(returns) > 1:
                metrics["volatility"] = float(returns.std())
                
                # Sharpe ratio (simplified, assuming risk-free rate of 2%)
                annual_return = returns.mean() * 252
                annual_volatility = returns.std() * (252 ** 0.5)
                metrics["sharpe_ratio"] = float((annual_return - 0.02) / annual_volatility) if annual_volatility > 0 else 0.0
                
                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                metrics["max_drawdown"] = float(drawdown.min())
                
                # VaR (Value at Risk) - 95% confidence
                metrics["var"] = float(returns.quantile(0.05))
            else:
                metrics.update({"volatility": 0.05, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "var": 0.0})
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {"volatility": 0.05, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "var": 0.0}
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume and liquidity metrics"""
        try:
            metrics = {}
            
            # Volume analysis
            if 'volume' in df.columns:
                volume = df['volume']
                avg_volume = volume.rolling(window=20).mean()
                current_volume = volume.iloc[-1]
                avg_volume_val = avg_volume.iloc[-1]
                
                metrics["volume_ratio"] = float(current_volume / avg_volume_val) if avg_volume_val > 0 else 1.0
                metrics["liquidity"] = float(current_volume * df.get('close', df.iloc[:, -1]).iloc[-1])
            elif 'Volume' in df.columns:
                volume = df['Volume']
                avg_volume = volume.rolling(window=20).mean()
                current_volume = volume.iloc[-1]
                avg_volume_val = avg_volume.iloc[-1]
                
                metrics["volume_ratio"] = float(current_volume / avg_volume_val) if avg_volume_val > 0 else 1.0
                metrics["liquidity"] = float(current_volume * df.get('Close', df.iloc[:, -1]).iloc[-1])
            else:
                metrics.update({"volume_ratio": 1.0, "liquidity": 500000.0})
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating volume metrics: {e}")
            return {"volume_ratio": 1.0, "liquidity": 500000.0}
    
    def _calculate_sentiment_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment-related metrics (simplified for now)"""
        try:
            # This would normally integrate with news API, social media sentiment, etc.
            # For now, we'll derive sentiment from price action
            
            if 'close' in df.columns:
                prices = df['close']
            elif 'Close' in df.columns:
                prices = df['Close']
            else:
                prices = df.iloc[:, -1]
            
            # Simple price momentum as sentiment proxy
            if len(prices) >= 10:
                recent_change = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10]
                sentiment = 0.5 + (recent_change * 2)  # Scale to roughly 0-1
                sentiment = max(0.0, min(1.0, sentiment))
            else:
                sentiment = 0.5
            
            return {
                "sentiment": float(sentiment),
                "fear_greed": float(sentiment * 100),
                "pattern_strength": float(abs(sentiment - 0.5) * 2)  # How far from neutral
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment metrics: {e}")
            return {"sentiment": 0.5, "fear_greed": 50.0, "pattern_strength": 0.0}
    
    def _run_monte_carlo_simulation(self, df: pd.DataFrame, capital: float) -> Dict[str, float]:
        """Run Monte Carlo simulation for probability estimation"""
        try:
            # Simplified Monte Carlo simulation
            if 'close' in df.columns:
                prices = df['close']
            elif 'Close' in df.columns:
                prices = df['Close']
            else:
                prices = df.iloc[:, -1]
            
            returns = prices.pct_change().dropna()
            
            if len(returns) < 10:
                return {"probabilidad_ganancia": 0.5, "roi_promedio": 0.0, "roi_std": 0.0}
            
            # Monte Carlo parameters
            num_simulations = 1000
            days_forward = 30
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Run simulations
            final_returns = []
            for _ in range(num_simulations):
                random_returns = pd.Series(
                    [mean_return + std_return * pd.Series([0]).sample(1, replace=True).iloc[0] 
                     for _ in range(days_forward)]
                )
                final_return = (1 + random_returns).prod() - 1
                final_returns.append(final_return)
            
            final_returns = pd.Series(final_returns)
            
            return {
                "probabilidad_ganancia": float((final_returns > 0).mean()),
                "roi_promedio": float(final_returns.mean()),
                "roi_std": float(final_returns.std())
            }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            return {"probabilidad_ganancia": 0.5, "roi_promedio": 0.0, "roi_std": 0.0}
    
    def _estimate_roi(self, metrics: Dict[str, Any]) -> float:
        """Estimate potential ROI based on metrics"""
        try:
            technical_score = metrics.get("technical_score", 0.5)
            risk_score = metrics.get("risk_score", 0.5)
            sentiment_score = metrics.get("sentiment_score", 0.5)
            
            # Simple ROI estimation
            base_roi = (technical_score + sentiment_score) / 2 - 0.5  # -0.5 to 0.5 range
            risk_adjusted_roi = base_roi * risk_score
            
            return float(risk_adjusted_roi)
            
        except Exception as e:
            self.logger.error(f"Error estimating ROI: {e}")
            return 0.0
    
    def _format_decision_result(self, decision_result, symbol: str, roi: float, 
                              mc_roi: float, metrics: Dict[str, Any], mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format DecisionResult into the expected system format"""
        return {
            # Standard DecisionMapper fields
            "action": decision_result.action,
            "sub_strategy": decision_result.sub_strategy,
            "confidence": decision_result.confidence,
            "reasons": decision_result.reasons,
            "alternatives": decision_result.alternatives,
            "explanation": decision_result.explanation,
            "votes": decision_result.votes,
            
            # Strategy metadata
            "strategy": self.name,
            "symbol": symbol,
            "profile": decision_result.profile,
            
            # ROI and metrics
            "roi": roi,
            "mc_roi": mc_roi,
            "roi_total": roi,
            "metrics": metrics,
            "montecarlo": mc_results,
            "metrics_used": decision_result.metrics_used,
            
            # Timestamps and status
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            
            # Additional strategy info
            "strategy_config": self.config,
            "decision_engine": "DecisionMapper"
        }
    
    def _enhance_result_with_strategy_data(self, result: Dict[str, Any], df: pd.DataFrame, 
                                         capital: float, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Add BolivarMode-specific enhancements to the result"""
        try:
            # Add position sizing recommendations
            if result["action"] in ["buy", "sell"]:
                position_size = min(self.config["max_position_size"], result["confidence"] * 0.15)
                result["position_size"] = position_size
                result["position_value"] = capital * position_size
                
                # Add stop loss and take profit levels
                latest_price = self._get_latest_price(df)
                if latest_price > 0:
                    if result["action"] == "buy":
                        result["stop_loss"] = latest_price * (1 - self.config["stop_loss_pct"])
                        result["take_profit"] = latest_price * (1 + self.config["take_profit_pct"])
                    else:  # sell
                        result["stop_loss"] = latest_price * (1 + self.config["stop_loss_pct"])
                        result["take_profit"] = latest_price * (1 - self.config["take_profit_pct"])
            
            # Add risk assessment
            result["risk_level"] = self._assess_risk_level(result["confidence"], result["votes"])
            
            # Add execution recommendations
            result["execution_strategy"] = self._recommend_execution_strategy(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error enhancing result: {e}")
            return result
    
    def _assess_risk_level(self, confidence: float, votes: Dict[str, float]) -> str:
        """Assess the risk level of the decision"""
        risk_score = votes.get("risk_assessment", 0.5)
        volatility_score = votes.get("volatility_analysis", 0.5)
        
        avg_risk = (risk_score + volatility_score) / 2
        
        if avg_risk > 0.7 and confidence > 0.7:
            return "low"
        elif avg_risk > 0.5 or confidence > 0.6:
            return "medium"
        else:
            return "high"
    
    def _recommend_execution_strategy(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Recommend execution strategy based on decision parameters"""
        confidence = result["confidence"]
        action = result["action"]
        
        if confidence > 0.8:
            order_type = "market"
            timing = "immediate"
        elif confidence > 0.6:
            order_type = "limit"
            timing = "opportunistic"
        else:
            order_type = "conditional"
            timing = "patient"
        
        return {
            "order_type": order_type,
            "timing": timing,
            "split_orders": confidence < 0.7,  # Split orders for lower confidence
            "monitoring_required": confidence < 0.8
        }
    
    def _create_error_result(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """Create a standardized error result"""
        return {
            "action": "hold",
            "sub_strategy": "error_fallback",
            "confidence": 0.0,
            "reasons": [f"Error in BolivarMode processing: {error_msg}"],
            "alternatives": [],
            "explanation": f"BolivarMode encountered an error: {error_msg}. Defaulting to hold for safety.",
            "votes": {},
            "strategy": self.name,
            "symbol": symbol,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "decision_engine": "DecisionMapper"
        }