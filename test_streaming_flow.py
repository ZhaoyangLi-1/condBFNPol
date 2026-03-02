#!/usr/bin/env python3
"""Test script to verify StreamingFlowPolicy implementation."""

import torch
import numpy as np

def test_streaming_flow_import():
    """Test that we can import the unified streaming flow policy."""
    try:
        from policies.streaming_flow_policy import StreamingFlowPolicy
        print("✓ StreamingFlowPolicy import successful")
        return True
    except ImportError as e:
        print(f"✗ StreamingFlowPolicy import failed: {e}")
        return False

def test_policy_instantiation():
    """Test policy instantiation with both image and low-dim data."""
    try:
        from policies.streaming_flow_policy import StreamingFlowPolicy
        
        # Create shape_meta with both images and low-dim data
        shape_meta = {
            "action": {"shape": [7]},
            "obs": {
                "camera_0": {"shape": [3, 480, 480], "type": "rgb"},
                "camera_1": {"shape": [3, 480, 480], "type": "rgb"},
                "robot_eef_pose": {"shape": [6], "type": "low_dim"}
            }
        }
        
        # Instantiate policy
        policy = StreamingFlowPolicy(
            action_space=None,
            shape_meta=shape_meta,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            sigma0=0.1,
            sigma1=0.1,
            device="cpu",
            dtype="float32",
            crop_shape=[384, 384],
            image_feature_dim=64,
        )
        
        print("✓ StreamingFlowPolicy instantiation successful")
        print(f"  - Action dim: {policy.action_dim}")
        print(f"  - Image encoders: {list(policy.image_encoders.keys())}")
        print(f"  - Low-dim keys: {policy.lowdim_keys}")
        print(f"  - Total obs dim: {policy.obs_dim}")
        return True, policy
        
    except Exception as e:
        print(f"✗ StreamingFlowPolicy instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_observation_processing():
    """Test observation processing with mixed data types."""
    try:
        success, policy = test_policy_instantiation()
        if not success:
            return False
            
        # Create mixed observations
        batch_size = 2
        obs = {
            "camera_0": torch.randn(batch_size, 3, 480, 480),
            "camera_1": torch.randn(batch_size, 3, 480, 480),
            "robot_eef_pose": torch.randn(batch_size, 6)
        }
        
        # Process observations
        encoded_obs = policy._process_observations(obs)
        
        print("✓ Observation processing successful")
        print(f"  - Input: images (2 cameras) + eef_pose (6D)")
        print(f"  - Output shape: {encoded_obs.shape}")
        print(f"  - Expected shape: ({batch_size}, {policy.obs_dim * policy.n_obs_steps})")
        
        return True
        
    except Exception as e:
        print(f"✗ Observation processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test policy forward pass."""
    try:
        success, policy = test_policy_instantiation()
        if not success:
            return False
            
        # Create mixed observations
        batch_size = 2
        obs = {
            "camera_0": torch.randn(batch_size, 3, 480, 480),
            "camera_1": torch.randn(batch_size, 3, 480, 480),
            "robot_eef_pose": torch.randn(batch_size, 6)
        }
        
        # Forward pass
        result = policy.forward(obs)
        
        print("✓ Forward pass successful")
        print(f"  - Output action shape: {result['action'].shape}")
        print(f"  - Expected shape: ({batch_size}, {policy.n_action_steps}, {policy.action_dim})")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation."""
    try:
        success, policy = test_policy_instantiation()
        if not success:
            return False
            
        # Create dummy batch
        batch_size = 2
        batch = {
            "obs": {
                "camera_0": torch.randn(batch_size, 3, 480, 480),
                "camera_1": torch.randn(batch_size, 3, 480, 480),
                "robot_eef_pose": torch.randn(batch_size, 6)
            },
            "action": torch.randn(batch_size, 16, 7)  # (batch, horizon, action_dim)
        }
        
        # Compute loss
        loss_dict = policy.compute_loss(batch)
        
        print("✓ Loss computation successful")
        print(f"  - Loss value: {loss_dict['loss'].item():.6f}")
        print(f"  - Loss requires grad: {loss_dict['loss'].requires_grad}")
        return True
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_only_mode():
    """Test with image-only observations (like the original config)."""
    try:
        from policies.streaming_flow_policy import StreamingFlowPolicy
        
        # Image-only shape_meta
        shape_meta = {
            "action": {"shape": [7]},
            "obs": {
                "camera_0": {"shape": [3, 480, 480], "type": "rgb"},
                "camera_1": {"shape": [3, 480, 480], "type": "rgb"}
            }
        }
        
        policy = StreamingFlowPolicy(
            action_space=None,
            shape_meta=shape_meta,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            sigma0=0.1,
            sigma1=0.1,
            device="cpu",
            dtype="float32",
            crop_shape=[384, 384],
            image_feature_dim=64,
        )
        
        # Test with image-only observations
        batch_size = 2
        obs = {
            "camera_0": torch.randn(batch_size, 3, 480, 480),
            "camera_1": torch.randn(batch_size, 3, 480, 480)
        }
        
        result = policy.forward(obs)
        
        print("✓ Image-only mode successful")
        print(f"  - Obs dim: {policy.obs_dim} (image features only)")
        print(f"  - Output shape: {result['action'].shape}")
        return True
        
    except Exception as e:
        print(f"✗ Image-only mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing StreamingFlowPolicy Implementation")
    print("=" * 70)
    
    tests = [
        test_streaming_flow_import,
        test_policy_instantiation,
        test_observation_processing,
        test_forward_pass,
        test_loss_computation,
        test_image_only_mode,
    ]
    
    results = []
    for test_func in tests:
        print(f"\n--- {test_func.__name__} ---")
        try:
            result = test_func()
            if isinstance(result, tuple):
                result = result[0]  # Take first element if tuple
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("Summary:")
    for test_func, result in zip(tests, results):
        status = "✓" if result else "✗"
        print(f"{status} {test_func.__name__}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Unified StreamingFlowPolicy implementation complete.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train_workspace.py --config-name=train_streaming_flow_pusht_real")
        print("2. The policy supports both image and low-dim observations")
        print("3. Image features: 2 cameras × 64 features = 128")
        print("4. Low-dim features: robot_eef_pose (6D) [optional, currently commented]")
        print("5. Current obs_dim: 128 (images only) or 134 (images + pose)")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()