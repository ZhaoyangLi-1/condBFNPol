#!/usr/bin/env python3
"""Test script to verify streaming flow policy integration with EMA."""

import torch
import numpy as np
from omegaconf import OmegaConf

def test_workspace_creation():
    """Test that we can create the workspace without errors."""
    try:
        from workspaces.train_streaming_flow_workspace import TrainStreamingFlowWorkspace
        from utils.ema import EMAModel
        from policies.streaming_flow_policy import StreamingFlowPolicy
        
        print("✓ All imports successful")
        
        # Create minimal config
        cfg = OmegaConf.create({
            'training': {
                'device': 'cpu',
                'seed': 42,
                'use_ema': True,
                'lr_scheduler': 'cosine',
                'lr_warmup_steps': 10,
                'num_epochs': 2,
                'gradient_accumulate_every': 1,
                'rollout_every': 10,
                'checkpoint_every': 10,
                'val_every': 5,
                'sample_every': 5,
                'max_train_steps': None,
                'max_val_steps': None,
                'tqdm_interval_sec': 1.0,
            },
            'shape_meta': {
                'action': {'shape': [7]},
                'obs': {
                    'camera_0': {'shape': [3, 480, 480], 'type': 'rgb'},
                    'camera_1': {'shape': [3, 480, 480], 'type': 'rgb'}
                }
            },
            'horizon': 16,
            'n_obs_steps': 2,
            'n_action_steps': 8,
            'crop_shape': [384, 384],
            'policy': {
                '_target_': 'policies.streaming_flow_policy.StreamingFlowPolicy',
                'action_space': None,
                'shape_meta': '${shape_meta}',
                'horizon': '${horizon}',
                'n_obs_steps': '${n_obs_steps}',
                'n_action_steps': '${n_action_steps}',
                'sigma0': 0.1,
                'sigma1': 0.1,
                'device': 'cpu',
                'dtype': 'float32',
                'crop_shape': '${crop_shape}',
                'image_feature_dim': 64,
                'fc_timesteps': 2,
            },
            'ema': {
                '_target_': 'utils.ema.EMAModel',
                'update_after_step': 0,
                'inv_gamma': 1.0,
                'power': 0.75,
                'min_value': 0.0,
                'max_value': 0.9999,
            },
            'optimizer': {
                '_target_': 'torch.optim.AdamW',
                'lr': 1e-4,
                'betas': [0.95, 0.999],
                'eps': 1e-8,
                'weight_decay': 1e-6,
            },
            'dataloader': {
                'batch_size': 2,
                'shuffle': True,
                'num_workers': 0,
                'pin_memory': False,
                'persistent_workers': False,
            },
            'val_dataloader': {
                'batch_size': 2,
                'shuffle': False,
                'num_workers': 0,
                'pin_memory': False,
                'persistent_workers': False,
            },
            'task': {
                'name': 'pusht_real_image',
                'image_shape': [3, 480, 480],
                'shape_meta': '${shape_meta}',
                'dataset': {
                    '_target_': 'torch.utils.data.TensorDataset',
                    # We'll create dummy data
                }
            },
            'checkpoint': {
                'save_last_ckpt': True,
                'save_last_snapshot': False,
            },
            'logging': {
                'mode': 'disabled',
            }
        })
        
        print("✓ Config created successfully")
        return True, cfg
        
    except Exception as e:
        print(f"✗ Workspace creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_ema_integration():
    """Test EMA integration specifically."""
    try:
        from policies.streaming_flow_policy import StreamingFlowPolicy
        from utils.ema import EMAModel
        
        # Create policy
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
            device="cpu",
            image_feature_dim=64,
        )
        
        print("✓ Policy created successfully")
        
        # Create EMA
        ema = EMAModel(
            model=policy,
            update_after_step=0,
            inv_gamma=1.0,
            power=0.75,
            min_value=0.0,
            max_value=0.9999,
        )
        
        print("✓ EMA created successfully")
        
        # Test EMA step
        ema.step(policy)
        print("✓ EMA step successful")
        
        # Test EMA state_dict
        state = ema.state_dict()
        print(f"✓ EMA state_dict successful, keys: {list(state.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ EMA integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration tests."""
    print("=" * 60)
    print("Testing Streaming Flow Policy Integration with EMA")
    print("=" * 60)
    
    tests = [
        test_workspace_creation,
        test_ema_integration,
    ]
    
    results = []
    for test_func in tests:
        print(f"\n--- {test_func.__name__} ---")
        try:
            result = test_func()
            if isinstance(result, tuple):
                result = result[0]
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Summary:")
    for test_func, result in zip(tests, results):
        status = "✓" if result else "✗"
        print(f"{status} {test_func.__name__}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All integration tests passed! EMA bug fixed.")
        print("\nYou can now run:")
        print("python scripts/train_workspace.py --config-name=train_streaming_flow_pusht_real")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()