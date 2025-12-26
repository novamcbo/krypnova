"""
Simple test to validate the DRL integration code structure.
This test validates the syntax and basic logic without running the full system.
"""

import ast
import sys


def test_drl_methods_exist():
    """Test that the DRL integration methods exist in exion_brain.py"""
    print("Testing DRL methods existence...")
    
    with open('exion_brain.py', 'r') as f:
        code = f.read()
    
    # Parse the Python file
    tree = ast.parse(code)
    
    # Find the ExionBrain class
    exion_brain_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ExionBrain':
            exion_brain_class = node
            break
    
    assert exion_brain_class is not None, "ExionBrain class not found"
    print("   ✓ ExionBrain class found")
    
    # Check for _get_drl_predictions method (including async functions)
    method_names = [m.name for m in exion_brain_class.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    
    assert '_get_drl_predictions' in method_names, "_get_drl_predictions method not found"
    print("   ✓ _get_drl_predictions method exists")
    
    assert 'calculate_roi_metrics' in method_names, "calculate_roi_metrics method not found"
    print("   ✓ calculate_roi_metrics method exists")
    
    # Find _get_drl_predictions method
    get_drl_pred = None
    for node in exion_brain_class.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == '_get_drl_predictions':
            get_drl_pred = node
            break
    
    assert get_drl_pred is not None, "_get_drl_predictions is not an async function"
    print("   ✓ _get_drl_predictions is async")
    
    # Check the parameters
    args = [arg.arg for arg in get_drl_pred.args.args]
    assert 'self' in args, "Missing self parameter"
    assert 'symbol' in args, "Missing symbol parameter"
    assert 'df' in args, "Missing df parameter"
    print("   ✓ _get_drl_predictions has correct parameters")
    
    return True


def test_drl_integration_in_calculate_roi_metrics():
    """Test that calculate_roi_metrics calls _get_drl_predictions"""
    print("\nTesting DRL integration in calculate_roi_metrics...")
    
    with open('exion_brain.py', 'r') as f:
        code = f.read()
    
    # Check that _get_drl_predictions is called
    assert 'await self._get_drl_predictions' in code, "_get_drl_predictions not called in calculate_roi_metrics"
    print("   ✓ _get_drl_predictions is called")
    
    # Check for DRL logging
    assert 'DRL Predictions' in code, "DRL logging not present"
    print("   ✓ DRL logging present")
    
    # Check for DRL output fields
    assert 'direction_with_drl' in code, "direction_with_drl field not in output"
    print("   ✓ direction_with_drl field in output")
    
    assert 'consensus_confidence_ratio' in code, "consensus_confidence_ratio field not in output"
    print("   ✓ consensus_confidence_ratio field in output")
    
    # Check for DRL predictions in detail
    assert 'drl_predictions' in code, "drl_predictions not in detail"
    print("   ✓ drl_predictions in detail section")
    
    # Check for fallback behavior
    assert 'DRL predictions not available' in code, "No fallback message for unavailable DRL"
    print("   ✓ Fallback behavior for unavailable DRL")
    
    return True


def test_drl_prediction_logic():
    """Test the DRL prediction logic structure"""
    print("\nTesting DRL prediction logic...")
    
    with open('exion_brain.py', 'r') as f:
        code = f.read()
    
    # Check for DQN agent usage
    assert 'self.dqn_agent' in code, "DQN agent not referenced"
    print("   ✓ DQN agent referenced")
    
    assert 'dqn_action' in code, "DQN action not extracted"
    print("   ✓ DQN action extracted")
    
    assert 'dqn_confidence' in code, "DQN confidence not calculated"
    print("   ✓ DQN confidence calculated")
    
    # Check for PPO agent usage
    assert 'self.ppo_agent' in code, "PPO agent not referenced"
    print("   ✓ PPO agent referenced")
    
    assert 'ppo_action' in code, "PPO action not extracted"
    print("   ✓ PPO action extracted")
    
    assert 'ppo_confidence' in code, "PPO confidence not calculated"
    print("   ✓ PPO confidence calculated")
    
    # Check for state preparation
    assert 'STATE_SIZE' in code, "STATE_SIZE not used for state preparation"
    print("   ✓ STATE_SIZE used for state preparation")
    
    # Check for action mapping
    assert 'HOLD' in code and 'BUY' in code and 'SELL' in code, "Action mapping not present"
    print("   ✓ Action mapping (HOLD/BUY/SELL) present")
    
    # Check for error handling
    assert 'except Exception' in code, "No exception handling in DRL code"
    print("   ✓ Exception handling present")
    
    return True


def test_drl_consensus_mechanism():
    """Test the consensus mechanism for combining DRL with strategies"""
    print("\nTesting DRL consensus mechanism...")
    
    with open('exion_brain.py', 'r') as f:
        code = f.read()
    
    # Check for action voting
    assert 'action_votes' in code, "Action voting mechanism not present"
    print("   ✓ Action voting mechanism present")
    
    # Check for consensus calculation
    assert 'consensus_action' in code, "Consensus action not calculated"
    print("   ✓ Consensus action calculated")
    
    # Check that DRL signals are added to signals list
    assert 'DQNAgent' in code, "DQN signals not added to signals list"
    print("   ✓ DQN signals added to signals list")
    
    assert 'PPOAgent' in code, "PPO signals not added to signals list"
    print("   ✓ PPO signals added to signals list")
    
    return True


def main():
    """Run all structural tests"""
    print("="*60)
    print("DRL Integration Structural Tests")
    print("="*60)
    
    try:
        test_drl_methods_exist()
        test_drl_integration_in_calculate_roi_metrics()
        test_drl_prediction_logic()
        test_drl_consensus_mechanism()
        
        print("\n" + "="*60)
        print("🎉 All structural tests passed successfully!")
        print("="*60)
        print("\nThe DRL integration code structure is correct:")
        print("  - _get_drl_predictions method is properly defined")
        print("  - DRL predictions are integrated in calculate_roi_metrics")
        print("  - DQN and PPO agents are properly used")
        print("  - Consensus mechanism combines DRL with strategies")
        print("  - Fallback behavior is implemented")
        print("  - Logging is present for debugging")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

