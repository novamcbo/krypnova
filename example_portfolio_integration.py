"""
Example usage of DecisionMapper with portfolio optimization integration.

This example demonstrates how the enhanced DecisionMapper provides holistic
trading decisions that consider both individual asset analysis and portfolio-level
optimization metrics.
"""

import asyncio
import json
import sys
from unittest.mock import Mock

# Mock the external dependencies
class MockPortfolioOptimizer:
    def __init__(self, symbols=None, historical_data=None):
        self.symbols = symbols or []
        self.historical_data = historical_data
    
    async def optimize_portfolio(self, risk_tolerance=0.5, expected_returns=None, covariance_matrix=None):
        """Mock portfolio optimization that returns reasonable test data."""
        if not self.symbols:
            return None
        
        # Create mock optimal weights - more realistic allocation
        if len(self.symbols) >= 3:
            optimal_weights = {
                'BTC': 0.40,  # 40% BTC
                'ETH': 0.35,  # 35% ETH  
                'SOL': 0.25   # 25% SOL
            }
        else:
            equal_weight = 1.0 / len(self.symbols)
            optimal_weights = {symbol: equal_weight for symbol in self.symbols}
        
        return {
            'optimal_weights': optimal_weights,
            'expected_return': 0.145,  # 14.5% expected return
            'volatility': 0.185,       # 18.5% volatility
            'sharpe_ratio': 0.78,      # Good Sharpe ratio
            'max_drawdown': 0.12       # 12% max drawdown
        }

# Mock the krypnova module
sys.modules['krypnova'] = Mock()
sys.modules['krypnova.ai'] = Mock()
sys.modules['krypnova.ai.portfolio_optimizer'] = Mock()
sys.modules['krypnova.ai.portfolio_optimizer'].PortfolioOptimizer = MockPortfolioOptimizer

from decision_mapper import DecisionMapper


async def example_portfolio_enhanced_decision():
    """
    Example showing how DecisionMapper integrates portfolio optimization
    into trading decisions with comprehensive explanations.
    """
    print("🎯 DecisionMapper Portfolio Integration Example\n")
    
    # Initialize DecisionMapper
    decision_mapper = DecisionMapper()
    
    # Example 1: Well-aligned decision (buy underweight asset)
    print("📈 Example 1: Buying underweight asset (SOL)")
    print("=" * 50)
    
    individual_analysis_sol = {
        'symbol': 'SOL',
        'decision': 'buy',
        'confidence': 0.82,
        'reason': 'Strong breakout pattern with high volume confirmation',
        'signals': {
            'signal_strength': 0.18,
            'sentiment_score': 0.35
        },
        'risk_metrics': {
            'risk_score': 28,
            'VaR': 0.03
        },
        'patterns_score': {'score': 0.73},
        'price': 145.50
    }
    
    portfolio = {
        'BTC': {'qty': 0.8, 'price': 45000},   # ~69% of portfolio
        'ETH': {'qty': 2.5, 'price': 3200},    # ~19% of portfolio  
        'SOL': {'qty': 5.0, 'price': 145}      # ~12% of portfolio (underweight)
    }
    
    context = {
        'risk_tolerance': 0.6,
        'expected_returns': {'BTC': 0.12, 'ETH': 0.15, 'SOL': 0.22},
        'user_id': 'example_user',
        'profile': 'moderate'
    }
    
    # Make enhanced decision
    enhanced_decision_sol = await decision_mapper.make_enhanced_decision(
        symbol='SOL',
        individual_analysis=individual_analysis_sol,
        portfolio=portfolio,
        context=context
    )
    
    print(f"Original Decision: {enhanced_decision_sol.get('original_decision', 'N/A')}")
    print(f"Enhanced Decision: {enhanced_decision_sol.get('decision', 'N/A')}")
    print(f"Confidence Change: {enhanced_decision_sol.get('original_confidence', 0):.1%} → {enhanced_decision_sol.get('confidence', 0):.1%}")
    print(f"Portfolio Influence: {enhanced_decision_sol.get('portfolio_influence', 'N/A')}")
    
    if enhanced_decision_sol.get('enhanced_explanation'):
        print(f"\n📋 Enhanced Explanation:\n{enhanced_decision_sol['enhanced_explanation']}")
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Misaligned decision (buy overweight asset)
    print("⚠️ Example 2: Buying overweight asset (BTC)")
    print("=" * 50)
    
    individual_analysis_btc = {
        'symbol': 'BTC',
        'decision': 'buy',
        'confidence': 0.78,
        'reason': 'Bullish divergence on RSI with support level hold',
        'signals': {
            'signal_strength': 0.16,
            'sentiment_score': 0.28
        },
        'risk_metrics': {
            'risk_score': 32,
            'VaR': 0.04
        },
        'patterns_score': {'score': 0.68},
        'price': 45000
    }
    
    # Make enhanced decision for overweight BTC
    enhanced_decision_btc = await decision_mapper.make_enhanced_decision(
        symbol='BTC',
        individual_analysis=individual_analysis_btc,
        portfolio=portfolio,
        context=context
    )
    
    print(f"Original Decision: {enhanced_decision_btc.get('original_decision', 'N/A')}")
    print(f"Enhanced Decision: {enhanced_decision_btc.get('decision', 'N/A')}")
    print(f"Confidence Change: {enhanced_decision_btc.get('original_confidence', 0):.1%} → {enhanced_decision_btc.get('confidence', 0):.1%}")
    print(f"Portfolio Influence: {enhanced_decision_btc.get('portfolio_influence', 'N/A')}")
    
    if enhanced_decision_btc.get('enhanced_explanation'):
        print(f"\n📋 Enhanced Explanation:\n{enhanced_decision_btc['enhanced_explanation']}")
    
    print("\n" + "="*80 + "\n")
    
    # Example 3: Portfolio health analysis only
    print("📊 Example 3: Portfolio Health Analysis")
    print("=" * 50)
    
    # Initialize portfolio optimizer
    decision_mapper._initialize_portfolio_optimizer(portfolio, context.get('historical_data'))
    
    # Analyze portfolio health
    portfolio_health = await decision_mapper.analyze_portfolio_health(portfolio, context)
    
    if portfolio_health:
        print("Portfolio Health Metrics:")
        print(f"• Expected Return: {portfolio_health.expected_return:.2%}")
        print(f"• Volatility: {portfolio_health.volatility:.2%}")
        print(f"• Sharpe Ratio: {portfolio_health.sharpe_ratio:.2f}")
        print(f"• Risk Score: {portfolio_health.risk_score:.1f}/100")
        print(f"• Diversification Score: {portfolio_health.diversification_score:.2f}")
        print(f"• Weight Deviation: {portfolio_health.weight_deviation:.2%}")
        
        print(f"\nCurrent vs Optimal Weights:")
        for symbol in portfolio_health.optimal_weights:
            current = portfolio_health.current_weights.get(symbol, 0)
            optimal = portfolio_health.optimal_weights[symbol]
            deviation = current - optimal
            status = "✅" if abs(deviation) < 0.05 else ("⚠️" if abs(deviation) < 0.15 else "🔴")
            print(f"• {symbol}: {current:.1%} vs {optimal:.1%} ({deviation:+.1%}) {status}")
        
        # Generate portfolio explanation
        portfolio_explanation = decision_mapper.generate_portfolio_explanation(portfolio_health)
        print(f"\n{portfolio_explanation}")
    
    print("\n🎉 DecisionMapper successfully integrates portfolio optimization into trading decisions!")
    print("   ✅ Provides comprehensive explanations")
    print("   ✅ Considers portfolio-level risk and diversification") 
    print("   ✅ Adjusts confidence based on portfolio alignment")
    print("   ✅ Recommends rebalancing when needed")
    print("   ✅ Maintains professional portfolio management standards")


async def example_integration_benefits():
    """Show the key benefits of the portfolio integration."""
    print("\n🌟 Key Benefits of Portfolio-Enhanced DecisionMapper:")
    print("=" * 60)
    
    benefits = [
        "🎯 Holistic Decision Making: Considers both individual signals and portfolio context",
        "📊 Risk Management: Prevents overconcentration in single assets",
        "⚖️ Optimal Allocation: Guides towards portfolio optimization targets",
        "📈 Professional Standards: Follows modern portfolio theory principles",
        "🔍 Transparent Explanations: Clear reasoning for every recommendation",
        "🛡️ Robust Error Handling: Graceful fallback when portfolio data unavailable",
        "🔄 Backward Compatibility: Integrates seamlessly with existing ExionBrain logic",
        "📋 Comprehensive Reporting: Detailed portfolio health assessments"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print(f"\n💡 This enhancement makes DecisionMapper more suitable for:")
    features = [
        "Professional portfolio management",
        "Risk-conscious trading strategies", 
        "Multi-asset portfolio optimization",
        "Institutional-grade decision making",
        "Regulatory compliance requirements",
        "Sophisticated risk management"
    ]
    
    for feature in features:
        print(f"   • {feature}")


if __name__ == "__main__":
    asyncio.run(example_portfolio_enhanced_decision())
    asyncio.run(example_integration_benefits())