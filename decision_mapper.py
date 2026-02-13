"""
DecisionMapper: Advanced decision engine for trading strategies
Provides standardized decision output format with profile-based scoring
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DecisionResult:
    """Standardized decision result format"""
    action: str
    sub_strategy: str
    confidence: float
    reasons: List[str]
    alternatives: List[Dict[str, Any]]
    explanation: str
    votes: Dict[str, float]
    metrics_used: Dict[str, Any]
    profile: str


class DecisionMapper:
    """
    Advanced decision mapping engine that processes metrics and user profiles
    to generate standardized trading decisions with comprehensive reasoning.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Action confidence thresholds by profile
        self.profile_thresholds = {
            "conservative": {
                "buy_threshold": 0.8,
                "sell_threshold": 0.75,
                "hold_threshold": 0.6
            },
            "moderate": {
                "buy_threshold": 0.7,
                "sell_threshold": 0.65,
                "hold_threshold": 0.5
            },
            "aggressive": {
                "buy_threshold": 0.6,
                "sell_threshold": 0.55,
                "hold_threshold": 0.4
            }
        }
        
        # Weight factors for different indicators by profile
        self.profile_weights = {
            "conservative": {
                "technical": 0.4,
                "risk": 0.4,
                "sentiment": 0.2,
                "volume": 0.3,
                "volatility": 0.5
            },
            "moderate": {
                "technical": 0.5,
                "risk": 0.3,
                "sentiment": 0.3,
                "volume": 0.4,
                "volatility": 0.3
            },
            "aggressive": {
                "technical": 0.6,
                "risk": 0.2,
                "sentiment": 0.4,
                "volume": 0.5,
                "volatility": 0.2
            }
        }
    
    def map_decision(self, user_profile: str, metrics: Dict[str, Any]) -> DecisionResult:
        """
        Core decision mapping method that processes all metrics and returns
        standardized decision with comprehensive reasoning.
        
        Args:
            user_profile: Risk profile (conservative, moderate, aggressive)
            metrics: Dictionary containing all available indicators and metrics
            
        Returns:
            DecisionResult with standardized format
        """
        try:
            self.logger.info(f"Mapping decision for profile: {user_profile}")
            
            # Normalize profile
            profile = user_profile.lower() if user_profile else "moderate"
            if profile not in self.profile_thresholds:
                profile = "moderate"
            
            # Extract and process indicators
            indicators = self._extract_indicators(metrics)
            
            # Calculate confidence scores for each action
            action_scores = self._calculate_action_scores(indicators, profile)
            
            # Determine primary action
            primary_action = self._select_primary_action(action_scores, profile)
            
            # Generate sub-strategy
            sub_strategy = self._determine_sub_strategy(primary_action, indicators, profile)
            
            # Calculate final confidence
            confidence = action_scores.get(primary_action, 0.0)
            
            # Generate comprehensive reasons
            reasons = self._generate_reasons(primary_action, indicators, profile)
            
            # Generate alternatives
            alternatives = self._generate_alternatives(action_scores, primary_action)
            
            # Create detailed explanation
            explanation = self._generate_explanation(
                primary_action, confidence, indicators, profile, reasons
            )
            
            # Prepare voting breakdown
            votes = {
                "technical_analysis": indicators.get("technical_score", 0.0),
                "risk_assessment": indicators.get("risk_score", 0.0),
                "sentiment_analysis": indicators.get("sentiment_score", 0.0),
                "volume_analysis": indicators.get("volume_score", 0.0),
                "volatility_analysis": indicators.get("volatility_score", 0.0),
                "montecarlo_simulation": indicators.get("monte_carlo_score", 0.0)
            }
            
            return DecisionResult(
                action=primary_action,
                sub_strategy=sub_strategy,
                confidence=round(confidence, 4),
                reasons=reasons,
                alternatives=alternatives,
                explanation=explanation,
                votes=votes,
                metrics_used=indicators,
                profile=profile
            )
            
        except Exception as e:
            self.logger.error(f"Error in map_decision: {e}", exc_info=True)
            return self._create_fallback_decision(user_profile, str(e))
    
    def _extract_indicators(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize all available indicators from metrics"""
        indicators = {}
        
        # Technical indicators
        indicators["rsi"] = self._safe_extract_numeric(metrics, "rsi", 50.0)
        indicators["macd"] = self._safe_extract_numeric(metrics, "macd", 0.0)
        indicators["bollinger_position"] = self._safe_extract_numeric(metrics, "bollinger_position", 0.5)
        indicators["sma_trend"] = self._safe_extract_numeric(metrics, "sma_trend", 0.0)
        indicators["ema_trend"] = self._safe_extract_numeric(metrics, "ema_trend", 0.0)
        
        # Risk metrics
        indicators["volatility"] = self._safe_extract_numeric(metrics, "volatility", 0.05)
        indicators["sharpe_ratio"] = self._safe_extract_numeric(metrics, "sharpe_ratio", 0.0)
        indicators["max_drawdown"] = self._safe_extract_numeric(metrics, "max_drawdown", 0.0)
        indicators["var"] = self._safe_extract_numeric(metrics, "var", 0.0)
        
        # Market sentiment
        indicators["sentiment"] = self._safe_extract_numeric(metrics, "sentiment", 0.5)
        indicators["fear_greed"] = self._safe_extract_numeric(metrics, "fear_greed", 50.0)
        
        # Volume and liquidity
        indicators["volume_ratio"] = self._safe_extract_numeric(metrics, "volume_ratio", 1.0)
        indicators["liquidity"] = self._safe_extract_numeric(metrics, "liquidity", 500000)
        
        # Pattern recognition
        indicators["pattern_strength"] = self._safe_extract_numeric(metrics, "pattern_strength", 0.0)
        indicators["support_resistance"] = self._safe_extract_numeric(metrics, "support_resistance", 0.5)
        
        # Monte Carlo results
        mc_data = metrics.get("montecarlo", {})
        indicators["monte_carlo_prob"] = self._safe_extract_numeric(mc_data, "probabilidad_ganancia", 0.5)
        indicators["monte_carlo_roi"] = self._safe_extract_numeric(mc_data, "roi_promedio", 0.0)
        
        # Calculate composite scores
        indicators["technical_score"] = self._calculate_technical_score(indicators)
        indicators["risk_score"] = self._calculate_risk_score(indicators)
        indicators["sentiment_score"] = self._normalize_sentiment(indicators["sentiment"])
        indicators["volume_score"] = self._calculate_volume_score(indicators)
        indicators["volatility_score"] = self._calculate_volatility_score(indicators["volatility"])
        indicators["monte_carlo_score"] = indicators["monte_carlo_prob"]
        
        return indicators
    
    def _safe_extract_numeric(self, data: Dict[str, Any], key: str, default: float) -> float:
        """Safely extract numeric value from nested dictionary"""
        try:
            value = data.get(key, default)
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value) if value.replace('.', '').replace('-', '').isdigit() else default
            return default
        except (ValueError, TypeError):
            return default
    
    def _calculate_technical_score(self, indicators: Dict[str, float]) -> float:
        """Calculate composite technical analysis score"""
        rsi = indicators["rsi"]
        macd = indicators["macd"]
        bollinger = indicators["bollinger_position"]
        sma_trend = indicators["sma_trend"]
        pattern = indicators["pattern_strength"]
        
        # RSI scoring (oversold/overbought) - more nuanced
        if rsi < 20:
            rsi_score = 0.9  # Very oversold - very bullish
        elif rsi < 30:
            rsi_score = 0.7  # Oversold - bullish
        elif rsi < 40:
            rsi_score = 0.6  # Moderately bullish
        elif rsi > 80:
            rsi_score = 0.1  # Very overbought - very bearish
        elif rsi > 70:
            rsi_score = 0.3  # Overbought - bearish
        elif rsi > 60:
            rsi_score = 0.4  # Moderately bearish
        else:
            rsi_score = 0.5  # Neutral range (40-60)
        
        # MACD scoring - more conservative
        macd_normalized = max(-0.1, min(0.1, macd))  # Clamp to reasonable range
        macd_score = 0.5 + (macd_normalized / 0.1) * 0.2  # Max influence of ±20%
        
        # Bollinger bands scoring - more conservative
        if bollinger > 0.8:
            bollinger_score = 0.3  # Near upper band - potentially overextended
        elif bollinger < 0.2:
            bollinger_score = 0.7  # Near lower band - potential bounce
        else:
            bollinger_score = bollinger  # Use as-is for middle range
        
        # Trend scoring - more conservative
        sma_clamped = max(-0.1, min(0.1, sma_trend))  # Clamp extreme values
        trend_score = 0.5 + (sma_clamped / 0.1) * 0.3  # Max influence of ±30%
        
        # Pattern strength - reduce influence
        pattern_score = 0.5 + (pattern - 0.5) * 0.4  # Center around neutral
        
        # Weighted average with more conservative weights
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        scores = [rsi_score, macd_score, bollinger_score, trend_score, pattern_score]
        
        technical_score = sum(w * s for w, s in zip(weights, scores))
        
        # Apply additional dampening to prevent extreme technical scores
        return max(0.2, min(0.8, technical_score))  # Keep between 20% and 80%
    
    def _calculate_risk_score(self, indicators: Dict[str, float]) -> float:
        """Calculate risk-adjusted score (higher is better for risk management)"""
        volatility = indicators["volatility"]
        sharpe = indicators["sharpe_ratio"]
        drawdown = abs(indicators["max_drawdown"])
        var = abs(indicators["var"])
        
        # Volatility scoring (lower volatility = higher score for conservative)
        vol_score = max(0, 1 - volatility * 10)
        
        # Sharpe ratio scoring
        sharpe_score = max(0, min(1, (sharpe + 2) / 4))
        
        # Drawdown scoring (lower drawdown = higher score)
        drawdown_score = max(0, 1 - drawdown)
        
        # VaR scoring (lower VaR = higher score)
        var_score = max(0, 1 - var)
        
        return (vol_score * 0.4 + sharpe_score * 0.3 + drawdown_score * 0.2 + var_score * 0.1)
    
    def _normalize_sentiment(self, sentiment: float) -> float:
        """Normalize sentiment to 0-1 range"""
        if sentiment >= 0 and sentiment <= 1:
            return sentiment
        elif sentiment >= -1 and sentiment < 0:
            return (sentiment + 1) / 2
        elif sentiment > 1:
            return min(1.0, sentiment / 100)  # Assume 0-100 scale
        return 0.5  # Default neutral
    
    def _calculate_volume_score(self, indicators: Dict[str, float]) -> float:
        """Calculate volume-based score"""
        volume_ratio = indicators["volume_ratio"]
        liquidity = indicators["liquidity"]
        
        # Volume ratio scoring (higher = better)
        vol_score = min(1.0, volume_ratio / 2.0)
        
        # Liquidity scoring (higher = better, normalized)
        liq_score = min(1.0, liquidity / 1000000)
        
        return (vol_score * 0.6 + liq_score * 0.4)
    
    def _calculate_volatility_score(self, volatility: float) -> float:
        """Calculate volatility score (context-dependent)"""
        # For most strategies, moderate volatility is preferred
        if volatility < 0.01:
            return 0.3  # Too low
        elif volatility > 0.1:
            return 0.2  # Too high
        else:
            return 1.0 - abs(volatility - 0.03) / 0.07
    
    def _calculate_action_scores(self, indicators: Dict[str, float], profile: str) -> Dict[str, float]:
        """Calculate confidence scores for each possible action"""
        weights = self.profile_weights[profile]
        
        # Base scores from indicators
        technical = indicators["technical_score"]
        risk = indicators["risk_score"]
        sentiment = indicators["sentiment_score"]
        volume = indicators["volume_score"]
        volatility = indicators["volatility_score"]
        monte_carlo = indicators["monte_carlo_score"]
        
        # Calculate weighted composite scores with normalization
        total_weight = sum([
            weights["technical"], weights["risk"], weights["sentiment"], 
            weights["volume"], weights["volatility"]
        ]) + 0.2  # Monte Carlo weight
        
        buy_score = (
            technical * weights["technical"] +
            risk * weights["risk"] * 0.8 +  # Risk adjustment for buy
            sentiment * weights["sentiment"] +
            volume * weights["volume"] +
            volatility * weights["volatility"] +
            monte_carlo * 0.2
        ) / total_weight  # Normalize by total weight
        
        sell_score = (
            (1 - technical) * weights["technical"] +
            risk * weights["risk"] +
            (1 - sentiment) * weights["sentiment"] +
            volume * weights["volume"] * 0.8 +  # Volume less critical for sell
            volatility * weights["volatility"] +
            (1 - monte_carlo) * 0.2
        ) / total_weight  # Normalize by total weight
        
        hold_score = (
            0.5 * weights["technical"] +
            risk * weights["risk"] * 1.2 +  # Risk important for hold
            0.5 * weights["sentiment"] +
            0.3 * weights["volume"] +
            volatility * weights["volatility"] * 0.5 +
            abs(monte_carlo - 0.5) * 0.1  # Neutral preference
        ) / (total_weight * 0.8)  # Slightly lower normalization for hold
        
        # Apply dampening factor to prevent overly confident scores
        dampening_factor = 0.85  # Max confidence will be 85%
        
        return {
            "buy": max(0.1, min(dampening_factor, buy_score)),    # Min 10%, max 85%
            "sell": max(0.1, min(dampening_factor, sell_score)),  # Min 10%, max 85%
            "hold": max(0.1, min(dampening_factor, hold_score))   # Min 10%, max 85%
        }
    
    def _select_primary_action(self, action_scores: Dict[str, float], profile: str) -> str:
        """Select the primary action based on scores and profile thresholds"""
        thresholds = self.profile_thresholds[profile]
        
        # Get the highest scoring action
        max_action = max(action_scores, key=action_scores.get)
        max_score = action_scores[max_action]
        
        # Apply profile-specific thresholds
        if max_action == "buy" and max_score >= thresholds["buy_threshold"]:
            return "buy"
        elif max_action == "sell" and max_score >= thresholds["sell_threshold"]:
            return "sell"
        elif max_score >= thresholds["hold_threshold"]:
            return max_action
        else:
            return "hold"  # Default to hold if no action meets threshold
    
    def _determine_sub_strategy(self, action: str, indicators: Dict[str, float], profile: str) -> str:
        """Determine the sub-strategy based on action and market conditions"""
        if action == "buy":
            if indicators["volatility"] > 0.05:
                return "volatility_breakout"
            elif indicators["monte_carlo_prob"] > 0.7:
                return "momentum_entry"
            elif indicators["rsi"] < 35:
                return "oversold_recovery"
            else:
                return "trend_following"
        
        elif action == "sell":
            if indicators["rsi"] > 70:
                return "profit_taking"
            elif indicators["volatility"] > 0.08:
                return "volatility_exit"
            elif indicators["technical_score"] < 0.3:
                return "technical_breakdown"
            else:
                return "risk_management"
        
        else:  # hold
            if indicators["volatility"] > 0.06:
                return "high_volatility_wait"
            elif abs(indicators["technical_score"] - 0.5) < 0.1:
                return "neutral_consolidation"
            else:
                return "strategic_patience"
    
    def _generate_reasons(self, action: str, indicators: Dict[str, float], profile: str) -> List[str]:
        """Generate detailed reasons for the decision"""
        reasons = []
        
        if action == "buy":
            if indicators["technical_score"] > 0.6:
                reasons.append(f"Technical indicators show bullish signals (score: {indicators['technical_score']:.2f})")
            if indicators["monte_carlo_prob"] > 0.6:
                reasons.append(f"Monte Carlo simulation indicates {indicators['monte_carlo_prob']*100:.1f}% probability of profit")
            if indicators["rsi"] < 35:
                reasons.append(f"RSI at {indicators['rsi']:.1f} suggests oversold conditions")
            if indicators["volume_score"] > 0.6:
                reasons.append("Strong volume support indicates institutional interest")
            
        elif action == "sell":
            if indicators["technical_score"] < 0.4:
                reasons.append(f"Technical indicators show bearish signals (score: {indicators['technical_score']:.2f})")
            if indicators["rsi"] > 70:
                reasons.append(f"RSI at {indicators['rsi']:.1f} indicates overbought conditions")
            if indicators["volatility"] > 0.08:
                reasons.append(f"High volatility ({indicators['volatility']*100:.1f}%) increases downside risk")
            
        else:  # hold
            reasons.append(f"Market conditions don't meet {profile} profile thresholds for action")
            if abs(indicators["technical_score"] - 0.5) < 0.15:
                reasons.append("Technical indicators show mixed signals")
            if indicators["volatility"] > 0.06:
                reasons.append("High volatility suggests waiting for better entry/exit points")
        
        if not reasons:
            reasons.append(f"Decision based on {profile} risk profile parameters")
            
        return reasons
    
    def _generate_alternatives(self, action_scores: Dict[str, float], primary_action: str) -> List[Dict[str, Any]]:
        """Generate alternative actions with their scores"""
        alternatives = []
        
        for action, score in action_scores.items():
            if action != primary_action:
                alternatives.append({
                    "action": action,
                    "confidence": round(score, 4),
                    "rationale": f"Alternative action with {score*100:.1f}% confidence"
                })
        
        return sorted(alternatives, key=lambda x: x["confidence"], reverse=True)
    
    def _generate_explanation(self, action: str, confidence: float, indicators: Dict[str, float], 
                            profile: str, reasons: List[str]) -> str:
        """Generate a comprehensive explanation of the decision"""
        explanation = f"Decision: {action.upper()} with {confidence*100:.1f}% confidence for {profile} profile.\n\n"
        explanation += "Key factors:\n"
        for reason in reasons:
            explanation += f"• {reason}\n"
        
        explanation += f"\nMarket analysis:\n"
        explanation += f"• Technical score: {indicators['technical_score']:.2f}\n"
        explanation += f"• Risk assessment: {indicators['risk_score']:.2f}\n"
        explanation += f"• Sentiment: {indicators['sentiment_score']:.2f}\n"
        explanation += f"• Volume strength: {indicators['volume_score']:.2f}\n"
        explanation += f"• Volatility: {indicators['volatility']*100:.1f}%\n"
        
        return explanation
    
    def _create_fallback_decision(self, profile: str, error_msg: str) -> DecisionResult:
        """Create a safe fallback decision when processing fails"""
        return DecisionResult(
            action="hold",
            sub_strategy="error_fallback",
            confidence=0.0,
            reasons=[f"Error in decision processing: {error_msg}", "Defaulting to hold for safety"],
            alternatives=[],
            explanation=f"An error occurred during decision mapping. Defaulting to hold position for safety. Error: {error_msg}",
            votes={},
            metrics_used={},
            profile=profile or "moderate"
        )