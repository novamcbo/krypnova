"""
Test script for DecisionMapper functionality with portfolio optimization integration.
"""

import asyncio
import sys
import os
import logging
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

# Mock the external dependencies
class MockPortfolioOptimizer:
    def __init__(self, symbols=None, historical_data=None):
        self.symbols = symbols or []
        self.historical_data = historical_data
    
    async def optimize_portfolio(self, risk_tolerance=0.5, expected_returns=None, covariance_matrix=None):
        """Mock portfolio optimization that returns reasonable test data."""
        if not self.symbols:
            return None
        
        # Create mock optimal weights (equal weight)
        equal_weight = 1.0 / len(self.symbols)
        optimal_weights = {symbol: equal_weight for symbol in self.symbols}
        
        return {
            'optimal_weights': optimal_weights,
            'expected_return': 0.12,  # 12% expected return
            'volatility': 0.18,       # 18% volatility
            'sharpe_ratio': 0.67,     # Moderate Sharpe ratio
            'max_drawdown': 0.15      # 15% max drawdown
        }

# Mock the krypnova module
sys.modules['krypnova'] = Mock()
sys.modules['krypnova.ai'] = Mock()
sys.modules['krypnova.ai.portfolio_optimizer'] = Mock()
sys.modules['krypnova.ai.portfolio_optimizer'].PortfolioOptimizer = MockPortfolioOptimizer

# Now import our DecisionMapper
from decision_mapper import DecisionMapper, PortfolioHealthMetrics

async def test_portfolio_health_analysis():
    """Test portfolio health analysis functionality."""
    print("🧪 Testing Portfolio Health Analysis...")
    
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    
    decision_mapper = DecisionMapper(logger)
    
    # Test portfolio data
    portfolio = {
        'BTC': {'qty': 0.5, 'price': 45000},
        'ETH': {'qty': 2.0, 'price': 3000},
        'SOL': {'qty': 10.0, 'price': 100}
    }
    
    context = {
        'risk_tolerance': 0.6,
        'expected_returns': {'BTC': 0.15, 'ETH': 0.12, 'SOL': 0.20},
        'historical_data': {}
    }
    
    # Initialize portfolio optimizer
    decision_mapper._initialize_portfolio_optimizer(portfolio, context.get('historical_data'))
    
    # Analyze portfolio health
    health_metrics = await decision_mapper.analyze_portfolio_health(portfolio, context)
    
    if health_metrics:
        print(f"✅ Portfolio Health Analysis successful:")
        print(f"   Expected Return: {health_metrics.expected_return:.2%}")
        print(f"   Volatility: {health_metrics.volatility:.2%}")
        print(f"   Sharpe Ratio: {health_metrics.sharpe_ratio:.2f}")
        print(f"   Risk Score: {health_metrics.risk_score:.1f}/100")
        print(f"   Diversification Score: {health_metrics.diversification_score:.2f}")
        print(f"   Weight Deviation: {health_metrics.weight_deviation:.2%}")
    else:
        print("❌ Portfolio Health Analysis failed")
        return False
    
    return True

async def test_portfolio_explanation():
    """Test portfolio explanation generation."""
    print("\n🧪 Testing Portfolio Explanation Generation...")
    
    # Create test portfolio health metrics
    health_metrics = PortfolioHealthMetrics(
        expected_return=0.15,
        volatility=0.20,
        sharpe_ratio=0.75,
        max_drawdown=0.12,
        diversification_score=0.65,
        risk_score=45.0,
        optimal_weights={'BTC': 0.4, 'ETH': 0.35, 'SOL': 0.25},
        current_weights={'BTC': 0.5, 'ETH': 0.3, 'SOL': 0.2},
        weight_deviation=0.15
    )
    
    decision_mapper = DecisionMapper()
    explanation = decision_mapper.generate_portfolio_explanation(health_metrics)
    
    print("✅ Portfolio Explanation:")
    print(explanation)
    return True

async def test_action_alignment():
    """Test portfolio-action alignment assessment."""
    print("\n🧪 Testing Portfolio-Action Alignment...")
    
    health_metrics = PortfolioHealthMetrics(
        expected_return=0.15,
        volatility=0.20,
        sharpe_ratio=0.75,
        max_drawdown=0.12,
        diversification_score=0.65,
        risk_score=45.0,
        optimal_weights={'BTC': 0.4, 'ETH': 0.35, 'SOL': 0.25},
        current_weights={'BTC': 0.5, 'ETH': 0.3, 'SOL': 0.2},  # BTC overweight, SOL underweight
        weight_deviation=0.15
    )
    
    decision_mapper = DecisionMapper()
    
    # Test BUY action for overweight asset (should be misaligned)
    buy_alignment = decision_mapper.assess_portfolio_action_alignment(
        "buy", "BTC", health_metrics, {}
    )
    print(f"✅ BUY BTC (overweight) alignment: {buy_alignment['alignment']}")
    print(f"   Recommendation: {buy_alignment['recommendation']}")
    print(f"   Explanation: {buy_alignment['explanation']}")
    
    # Test BUY action for underweight asset (should be aligned)
    buy_sol_alignment = decision_mapper.assess_portfolio_action_alignment(
        "buy", "SOL", health_metrics, {}
    )
    print(f"\n✅ BUY SOL (underweight) alignment: {buy_sol_alignment['alignment']}")
    print(f"   Recommendation: {buy_sol_alignment['recommendation']}")
    print(f"   Explanation: {buy_sol_alignment['explanation']}")
    
    return True

async def test_enhanced_decision():
    """Test complete enhanced decision making."""
    print("\n🧪 Testing Enhanced Decision Making...")
    
    logger = logging.getLogger("test_decision")
    logger.setLevel(logging.INFO)
    
    decision_mapper = DecisionMapper(logger)
    
    # Mock individual analysis
    individual_analysis = {
        'symbol': 'BTC',
        'decision': 'buy',
        'confidence': 0.75,
        'reason': 'Strong technical signals and positive momentum',
        'signals': {'signal_strength': 0.15},
        'risk_metrics': {'risk_score': 35}
    }
    
    # Test portfolio
    portfolio = {
        'BTC': {'qty': 0.8, 'price': 45000},  # Overweight position
        'ETH': {'qty': 2.0, 'price': 3000},
        'SOL': {'qty': 5.0, 'price': 100}
    }
    
    context = {
        'risk_tolerance': 0.6,
        'expected_returns': {'BTC': 0.15, 'ETH': 0.12, 'SOL': 0.20},
        'historical_data': {}
    }
    
    # Make enhanced decision
    enhanced_result = await decision_mapper.make_enhanced_decision(
        symbol='BTC',
        individual_analysis=individual_analysis,
        portfolio=portfolio,
        context=context
    )
    
    print(f"✅ Enhanced Decision Results:")
    print(f"   Original Decision: {enhanced_result.get('original_decision')}")
    print(f"   Final Decision: {enhanced_result.get('decision')}")
    print(f"   Original Confidence: {enhanced_result.get('original_confidence', 0):.2%}")
    print(f"   Adjusted Confidence: {enhanced_result.get('confidence', 0):.2%}")
    print(f"   Portfolio Influence: {enhanced_result.get('portfolio_influence')}")
    print(f"   Portfolio Enhanced: {enhanced_result.get('portfolio_enhanced')}")
    
    if enhanced_result.get('enhanced_explanation'):
        print(f"\n📋 Enhanced Explanation:")
        print(enhanced_result['enhanced_explanation'])
    
    return True

async def main():
    """Run all tests."""
    print("🚀 Starting DecisionMapper Tests...\n")
    
    tests = [
        test_portfolio_health_analysis,
        test_portfolio_explanation,
        test_action_alignment,
        test_enhanced_decision
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
                print("✅ PASSED\n")
            else:
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ FAILED with error: {e}\n")
    
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DecisionMapper is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)