#!/usr/bin/env python3
"""Test script to verify streaming flow policy integration."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_streaming_flow_policy_import():
    """Test if streaming flow policy can be imported."""
    try:
        from policies.streaming_flow_policy import StreamingFlowPolicy
        print("✓ StreamingFlowPolicy import successful")
        return True
    except ImportError as e:
        print(f"✗ StreamingFlowPolicy import failed: {e}")
        return False

def test_streaming_flow_policy_creation():
    """Test if streaming flow policy can be created."""
    try:
        from policies.streaming_flow_policy import StreamingFlowPolicy
        
        # Mock action space (simplified)
        class MockActionSpace:
            def __init__(self, shape):
                self.shape = shape
                self.low = np.ones(shape) * -1
                self.high = np.ones(shape) * 1
        
        action_space = MockActionSpace((7,))
        
        shape_meta = {
            "action": {"shape": [7]},
            "obs": {
                "camera_0": {"shape": [3, 480, 480], "type": "rgb"},
                "camera_1": {"shape": [3, 480, 480], "type": "rgb"}
            }
        }
        
        policy = StreamingFlowPolicy(
            action_space=action_space,
            shape_meta=shape_meta,
            device="cpu"
        )
        
        print("✓ StreamingFlowPolicy creation successful")
        return True, policy
    except Exception as e:
        print(f"✗ StreamingFlowPolicy creation failed: {e}")
        return False, None

def test_streaming_flow_policy_inference():
    """Test if streaming flow policy can run inference."""
    success, policy = test_streaming_flow_policy_creation()
    if not success:
        return False
        
    try:
        # Create dummy observation
        obs = {
            "camera_0": torch.randn(1, 3, 480, 480),
            "camera_1": torch.randn(1, 3, 480, 480)
        }
        
        # Test forward pass
        with torch.no_grad():
            result = policy.forward(obs)
            
        assert "action" in result
        actions = result["action"]
        assert actions.shape[0] == 1  # batch size
        assert actions.shape[1] == policy.n_action_steps  # action steps
        assert actions.shape[2] == policy.action_dim  # action dimension
        
        print("✓ StreamingFlowPolicy inference successful")
        print(f"  Output shape: {actions.shape}")
        return True
    except Exception as e:
        print(f"✗ StreamingFlowPolicy inference failed: {e}")
        return False

def test_predict_action():
    """Test predict_action method."""
    success, policy = test_streaming_flow_policy_creation()
    if not success:
        return False
        
    try:
        # Create dummy observation with numpy arrays
        obs = {
            "camera_0": np.random.randn(3, 480, 480),
            "camera_1": np.random.randn(3, 480, 480)
        }
        
        # Test predict_action
        result = policy.predict_action(obs)
        
        assert "action" in result
        actions = result["action"]
        assert isinstance(actions, np.ndarray)
        assert actions.shape[0] == policy.n_action_steps
        assert actions.shape[1] == policy.action_dim
        
        print("✓ predict_action successful")
        print(f"  Output shape: {actions.shape}")
        return True
    except Exception as e:
        print(f"✗ predict_action failed: {e}")
        return False

def test_workspace_import():
    """Test if workspace can be imported."""
    try:
        from workspaces.train_streaming_flow_workspace import TrainStreamingFlowWorkspace
        print("✓ TrainStreamingFlowWorkspace import successful")
        return True
    except ImportError as e:
        print(f"✗ TrainStreamingFlowWorkspace import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Streaming Flow Policy Integration")
    print("=" * 50)
    
    tests = [
        test_streaming_flow_policy_import,
        test_streaming_flow_policy_creation,
        test_streaming_flow_policy_inference,
        test_predict_action,
        test_workspace_import,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        success = test()
        if success or success is True:  # Handle both bool and tuple returns
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration successful!")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())