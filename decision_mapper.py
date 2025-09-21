"""
DecisionMapper with integrated PortfolioOptimizer for holistic trading decisions.

This module provides explanations and trade action selection based on risk, indicators, 
patterns, defense, backtest, and portfolio-level optimization metrics.
"""

import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from krypnova.ai.portfolio_optimizer import PortfolioOptimizer


@dataclass
class PortfolioHealthMetrics:
    """Portfolio health assessment metrics."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_score: float
    risk_score: float
    optimal_weights: Dict[str, float]
    current_weights: Dict[str, float]
    weight_deviation: float


class DecisionMapper:
    """
    Enhanced decision mapper that integrates portfolio optimization into trading decisions.
    
    Provides holistic explanations and recommendations by considering:
    - Individual asset analysis (signals, patterns, risk)
    - Portfolio-level metrics and optimization
    - Risk management and diversification
    - Alignment between asset-level and portfolio-level recommendations
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the DecisionMapper with portfolio optimization capabilities."""
        self.logger = logger or logging.getLogger(__name__)
        self.portfolio_optimizer = None
    
    def _initialize_portfolio_optimizer(self, portfolio_data: Dict[str, Any], 
                                      historical_data: Optional[Dict] = None) -> bool:
        """Initialize portfolio optimizer with current portfolio data."""
        try:
            if not portfolio_data:
                return False
                
            self.portfolio_optimizer = PortfolioOptimizer(
                symbols=list(portfolio_data.keys()),
                historical_data=historical_data
            )
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize PortfolioOptimizer: {e}")
            return False
    
    async def analyze_portfolio_health(self, portfolio: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Optional[PortfolioHealthMetrics]:
        """
        Analyze current portfolio health and optimization metrics.
        
        Args:
            portfolio: Current portfolio holdings {symbol: {qty, price}}
            context: Trading context with historical data
            
        Returns:
            PortfolioHealthMetrics or None if analysis fails
        """
        try:
            if not portfolio or not self.portfolio_optimizer:
                return None
            
            # Extract current holdings and values
            current_weights = {}
            total_value = 0
            
            for symbol, holding in portfolio.items():
                if isinstance(holding, dict):
                    qty = holding.get('qty', 0)
                    price = holding.get('price', 0)
                    value = qty * price
                    total_value += value
                    current_weights[symbol] = value
            
            if total_value <= 0:
                return None
            
            # Normalize weights
            current_weights = {k: v / total_value for k, v in current_weights.items()}
            
            # Get optimization results
            optimization_result = await self.portfolio_optimizer.optimize_portfolio(
                risk_tolerance=context.get('risk_tolerance', 0.5),
                expected_returns=context.get('expected_returns', {}),
                covariance_matrix=context.get('covariance_matrix')
            )
            
            if not optimization_result:
                return None
            
            optimal_weights = optimization_result.get('optimal_weights', {})
            expected_return = optimization_result.get('expected_return', 0)
            volatility = optimization_result.get('volatility', 0)
            sharpe_ratio = optimization_result.get('sharpe_ratio', 0)
            
            # Calculate weight deviation
            weight_deviation = sum(
                abs(current_weights.get(symbol, 0) - optimal_weights.get(symbol, 0))
                for symbol in set(list(current_weights.keys()) + list(optimal_weights.keys()))
            ) / 2
            
            # Calculate diversification score (1 - Herfindahl index)
            herfindahl_index = sum(w ** 2 for w in current_weights.values())
            diversification_score = 1 - herfindahl_index
            
            # Calculate risk score based on various factors
            risk_score = self._calculate_portfolio_risk_score(
                volatility, sharpe_ratio, weight_deviation, diversification_score
            )
            
            return PortfolioHealthMetrics(
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=optimization_result.get('max_drawdown', 0),
                diversification_score=diversification_score,
                risk_score=risk_score,
                optimal_weights=optimal_weights,
                current_weights=current_weights,
                weight_deviation=weight_deviation
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio health: {e}")
            return None
    
    def _calculate_portfolio_risk_score(self, volatility: float, sharpe_ratio: float,
                                      weight_deviation: float, diversification_score: float) -> float:
        """Calculate overall portfolio risk score (0-100, higher = riskier)."""
        try:
            # Normalize components to 0-1 scale
            vol_risk = min(volatility * 4, 1.0)  # Assume vol > 0.25 is high risk
            sharpe_risk = max(0, 1 - sharpe_ratio / 2)  # Sharpe < 2 increases risk
            deviation_risk = min(weight_deviation * 2, 1.0)  # Deviation > 0.5 is high risk
            concentration_risk = 1 - diversification_score
            
            # Weighted combination
            risk_score = (
                0.3 * vol_risk +
                0.3 * sharpe_risk +
                0.2 * deviation_risk +
                0.2 * concentration_risk
            ) * 100
            
            return min(max(risk_score, 0), 100)
        except Exception:
            return 50.0  # Default moderate risk
    
    def generate_portfolio_explanation(self, portfolio_health: Optional[PortfolioHealthMetrics]) -> str:
        """Generate human-readable portfolio health explanation."""
        if not portfolio_health:
            return "⚠️ Portfolio data unavailable - individual asset analysis only."
        
        explanation_parts = ["📊 **Portfolio Health Assessment:**"]
        
        # Expected return assessment
        if portfolio_health.expected_return > 0.15:
            explanation_parts.append(f"✅ High expected return ({portfolio_health.expected_return:.1%})")
        elif portfolio_health.expected_return > 0.08:
            explanation_parts.append(f"📈 Moderate expected return ({portfolio_health.expected_return:.1%})")
        else:
            explanation_parts.append(f"⚠️ Low expected return ({portfolio_health.expected_return:.1%})")
        
        # Volatility assessment
        if portfolio_health.volatility > 0.25:
            explanation_parts.append(f"🔴 High volatility ({portfolio_health.volatility:.1%}) - Consider risk reduction")
        elif portfolio_health.volatility > 0.15:
            explanation_parts.append(f"🟡 Moderate volatility ({portfolio_health.volatility:.1%})")
        else:
            explanation_parts.append(f"🟢 Low volatility ({portfolio_health.volatility:.1%})")
        
        # Sharpe ratio assessment
        if portfolio_health.sharpe_ratio > 1.5:
            explanation_parts.append(f"🎯 Excellent risk-adjusted returns (Sharpe: {portfolio_health.sharpe_ratio:.2f})")
        elif portfolio_health.sharpe_ratio > 0.8:
            explanation_parts.append(f"📊 Good risk-adjusted returns (Sharpe: {portfolio_health.sharpe_ratio:.2f})")
        else:
            explanation_parts.append(f"⚠️ Poor risk-adjusted returns (Sharpe: {portfolio_health.sharpe_ratio:.2f})")
        
        # Diversification assessment
        if portfolio_health.diversification_score > 0.7:
            explanation_parts.append("🌐 Well-diversified portfolio")
        elif portfolio_health.diversification_score > 0.4:
            explanation_parts.append("📈 Moderately diversified portfolio")
        else:
            explanation_parts.append("🔴 Concentrated portfolio - Consider diversification")
        
        # Weight deviation assessment
        if portfolio_health.weight_deviation > 0.3:
            explanation_parts.append(f"⚖️ Significant deviation from optimal weights ({portfolio_health.weight_deviation:.1%}) - Rebalancing recommended")
        elif portfolio_health.weight_deviation > 0.15:
            explanation_parts.append(f"📊 Moderate deviation from optimal weights ({portfolio_health.weight_deviation:.1%})")
        else:
            explanation_parts.append("✅ Portfolio allocation close to optimal")
        
        return "\n".join(explanation_parts)
    
    def assess_portfolio_action_alignment(self, proposed_action: str, symbol: str,
                                        portfolio_health: Optional[PortfolioHealthMetrics],
                                        individual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess how the proposed action aligns with portfolio optimization recommendations.
        
        Returns:
            Dict with alignment assessment and recommendations
        """
        if not portfolio_health:
            return {
                "alignment": "unknown",
                "recommendation": "proceed_with_caution",
                "explanation": "Portfolio optimization unavailable - proceed based on individual analysis only."
            }
        
        current_weight = portfolio_health.current_weights.get(symbol, 0)
        optimal_weight = portfolio_health.optimal_weights.get(symbol, 0)
        weight_diff = optimal_weight - current_weight
        
        alignment_assessment = {
            "alignment": "neutral",
            "recommendation": "proceed",
            "explanation": "",
            "portfolio_suggestion": "",
            "risk_impact": "neutral"
        }
        
        # Analyze action alignment
        if proposed_action.lower() in ["buy", "long", "long_buy"]:
            if weight_diff > 0.05:  # Should increase position
                alignment_assessment.update({
                    "alignment": "strongly_aligned",
                    "recommendation": "proceed",
                    "explanation": f"✅ BUY aligns with portfolio optimization: current weight {current_weight:.1%} vs optimal {optimal_weight:.1%}",
                    "portfolio_suggestion": f"Consider increasing {symbol} position by {weight_diff:.1%} of portfolio value"
                })
            elif weight_diff > -0.05:  # Near optimal
                alignment_assessment.update({
                    "alignment": "aligned",
                    "recommendation": "proceed_cautiously",
                    "explanation": f"📊 BUY acceptable: {symbol} weight near optimal ({current_weight:.1%} vs {optimal_weight:.1%})",
                    "portfolio_suggestion": "Small position increase acceptable"
                })
            else:  # Should decrease position
                alignment_assessment.update({
                    "alignment": "misaligned",
                    "recommendation": "reconsider",
                    "explanation": f"⚠️ BUY conflicts with portfolio optimization: {symbol} overweight ({current_weight:.1%} vs optimal {optimal_weight:.1%})",
                    "portfolio_suggestion": f"Consider reducing {symbol} exposure instead",
                    "risk_impact": "negative"
                })
        
        elif proposed_action.lower() in ["sell", "short", "short_sell"]:
            if weight_diff < -0.05:  # Should decrease position
                alignment_assessment.update({
                    "alignment": "strongly_aligned",
                    "recommendation": "proceed",
                    "explanation": f"✅ SELL aligns with portfolio optimization: {symbol} overweight ({current_weight:.1%} vs optimal {optimal_weight:.1%})",
                    "portfolio_suggestion": f"Consider reducing {symbol} position by {abs(weight_diff):.1%} of portfolio value"
                })
            elif weight_diff < 0.05:  # Near optimal
                alignment_assessment.update({
                    "alignment": "aligned",
                    "recommendation": "proceed_cautiously",
                    "explanation": f"📊 SELL acceptable: {symbol} weight near optimal ({current_weight:.1%} vs {optimal_weight:.1%})",
                    "portfolio_suggestion": "Small position decrease acceptable"
                })
            else:  # Should increase position
                alignment_assessment.update({
                    "alignment": "misaligned",
                    "recommendation": "reconsider",
                    "explanation": f"⚠️ SELL conflicts with portfolio optimization: {symbol} underweight ({current_weight:.1%} vs optimal {optimal_weight:.1%})",
                    "portfolio_suggestion": f"Consider increasing {symbol} exposure instead",
                    "risk_impact": "negative"
                })
        
        # Add overall portfolio risk considerations
        if portfolio_health.risk_score > 80:
            alignment_assessment["risk_impact"] = "negative"
            alignment_assessment["explanation"] += f"\n🔴 High portfolio risk ({portfolio_health.risk_score:.0f}/100) - Consider risk reduction"
        elif portfolio_health.risk_score > 60:
            alignment_assessment["risk_impact"] = "caution"
            alignment_assessment["explanation"] += f"\n🟡 Moderate portfolio risk ({portfolio_health.risk_score:.0f}/100)"
        
        return alignment_assessment
    
    async def make_enhanced_decision(self, symbol: str, individual_analysis: Dict[str, Any],
                                   portfolio: Optional[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make an enhanced trading decision incorporating portfolio optimization.
        
        Args:
            symbol: Trading symbol
            individual_analysis: Individual asset analysis results
            portfolio: Current portfolio holdings
            context: Trading context and parameters
            
        Returns:
            Enhanced decision with portfolio-informed recommendations
        """
        try:
            # Initialize portfolio optimizer if portfolio data available
            portfolio_optimizer_available = False
            if portfolio:
                portfolio_optimizer_available = self._initialize_portfolio_optimizer(
                    portfolio, context.get('historical_data')
                )
            
            # Analyze portfolio health
            portfolio_health = None
            if portfolio_optimizer_available:
                portfolio_health = await self.analyze_portfolio_health(portfolio, context)
            
            # Get base decision from individual analysis
            base_decision = individual_analysis.get("decision", "hold")
            base_confidence = individual_analysis.get("confidence", 0.5)
            base_reason = individual_analysis.get("reason", "Based on individual asset analysis")
            
            # Generate portfolio explanation
            portfolio_explanation = self.generate_portfolio_explanation(portfolio_health)
            
            # Assess portfolio-action alignment
            alignment_assessment = self.assess_portfolio_action_alignment(
                base_decision, symbol, portfolio_health, individual_analysis
            )
            
            # Adjust decision based on portfolio alignment
            final_decision = base_decision
            adjusted_confidence = base_confidence
            portfolio_influence = "none"
            
            if alignment_assessment["alignment"] == "misaligned":
                if alignment_assessment["recommendation"] == "reconsider":
                    # Reduce confidence for misaligned actions
                    adjusted_confidence *= 0.6
                    portfolio_influence = "negative"
                    
                    # Consider alternative action if severely misaligned
                    if adjusted_confidence < 0.4:
                        if base_decision.lower() in ["buy", "long"]:
                            final_decision = "hold"
                        elif base_decision.lower() in ["sell", "short"]:
                            final_decision = "hold"
                        portfolio_influence = "override"
            
            elif alignment_assessment["alignment"] == "strongly_aligned":
                # Boost confidence for well-aligned actions
                adjusted_confidence = min(adjusted_confidence * 1.2, 1.0)
                portfolio_influence = "positive"
            
            # Generate comprehensive explanation
            enhanced_explanation = self._generate_enhanced_explanation(
                individual_analysis, portfolio_explanation, alignment_assessment,
                base_decision, final_decision, portfolio_influence
            )
            
            # Build enhanced result
            enhanced_result = {
                **individual_analysis,  # Preserve original analysis
                "decision": final_decision,
                "confidence": adjusted_confidence,
                "portfolio_enhanced": True,
                "portfolio_health": portfolio_health.__dict__ if portfolio_health else None,
                "portfolio_alignment": alignment_assessment,
                "portfolio_influence": portfolio_influence,
                "enhanced_explanation": enhanced_explanation,
                "original_decision": base_decision,
                "original_confidence": base_confidence,
                "portfolio_explanation": portfolio_explanation
            }
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced decision making: {e}\n{traceback.format_exc()}")
            # Return original analysis with error note
            return {
                **individual_analysis,
                "portfolio_enhanced": False,
                "portfolio_error": str(e),
                "enhanced_explanation": f"Portfolio analysis failed: {e}. Decision based on individual analysis only."
            }
    
    def _generate_enhanced_explanation(self, individual_analysis: Dict[str, Any],
                                     portfolio_explanation: str, alignment_assessment: Dict[str, Any],
                                     base_decision: str, final_decision: str,
                                     portfolio_influence: str) -> str:
        """Generate comprehensive explanation combining individual and portfolio analysis."""
        explanation_parts = []
        
        # Individual analysis summary
        explanation_parts.append("🔍 **Individual Asset Analysis:**")
        base_reason = individual_analysis.get("reason", "No specific reason provided")
        explanation_parts.append(f"• Base recommendation: {base_decision.upper()}")
        explanation_parts.append(f"• Reasoning: {base_reason}")
        
        # Add individual metrics if available
        if "signals" in individual_analysis:
            signals = individual_analysis["signals"]
            if "signal_strength" in signals:
                explanation_parts.append(f"• Signal strength: {signals['signal_strength']:.2%}")
        
        if "risk_metrics" in individual_analysis:
            risk_metrics = individual_analysis["risk_metrics"]
            if "risk_score" in risk_metrics:
                explanation_parts.append(f"• Risk score: {risk_metrics['risk_score']}")
        
        # Portfolio analysis
        explanation_parts.append("\n" + portfolio_explanation)
        
        # Alignment assessment
        explanation_parts.append("\n🔄 **Portfolio Alignment Analysis:**")
        explanation_parts.append(alignment_assessment["explanation"])
        
        if alignment_assessment.get("portfolio_suggestion"):
            explanation_parts.append(f"• Suggestion: {alignment_assessment['portfolio_suggestion']}")
        
        # Final decision explanation
        explanation_parts.append("\n📋 **Final Recommendation:**")
        
        if portfolio_influence == "override":
            explanation_parts.append(f"• Decision changed from {base_decision.upper()} to {final_decision.upper()} due to portfolio misalignment")
        elif portfolio_influence == "positive":
            explanation_parts.append(f"• Decision {final_decision.upper()} reinforced by portfolio optimization")
        elif portfolio_influence == "negative":
            explanation_parts.append(f"• Decision {final_decision.upper()} confidence reduced due to portfolio concerns")
        else:
            explanation_parts.append(f"• Decision {final_decision.upper()} based primarily on individual analysis")
        
        return "\n".join(explanation_parts)


# Helper function for backward compatibility
async def make_portfolio_enhanced_decision(symbol: str, individual_analysis: Dict[str, Any],
                                         portfolio: Optional[Dict[str, Any]], context: Dict[str, Any],
                                         logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Convenience function for making portfolio-enhanced decisions.
    
    This function can be used as a drop-in replacement for existing decision logic.
    """
    decision_mapper = DecisionMapper(logger)
    return await decision_mapper.make_enhanced_decision(symbol, individual_analysis, portfolio, context)