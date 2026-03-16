#!/usr/bin/env python3
"""Test streaming flow velocity function only."""

import sys
import pathlib
import torch

# Add project root to path
project_root = pathlib.Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_velocity_function():
    """Test the velocity function directly."""
    print("Testing velocity function...")
    
    try:
        import hydra
        from hydra import initialize, compose
        
        # Initialize Hydra with config directory
        with initialize(version_base=None, config_path="config"):
            # Load the streaming flow config and modify for testing
            cfg = compose(config_name="train_streaming_flow_pusht.yaml", 
                         overrides=[
                             "training.device=cpu",
                             "dataloader.batch_size=1",
                         ])
            
            print(f"✓ Config loaded")
            
            # Get workspace class and instantiate
            workspace_cls = hydra.utils.get_class(cfg._target_)
            
            # Create a temporary output directory
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = pathlib.Path(tmpdir)
                
                try:
                    workspace = workspace_cls(cfg, output_dir=output_dir)
                    policy = workspace.policy
                    print(f"✓ Policy created: {type(policy).__name__}")
                    
                    # Test velocity function directly
                    batch_size = 1
                    action_dim = 2
                    horizon = 16
                    obs_dim = 55332  # From the error message
                    
                    # Create test input
                    x = torch.randn(batch_size, action_dim * horizon + obs_dim)
                    t = torch.tensor([0.5])
                    
                    print(f"✓ Input tensor shape: {x.shape}")
                    print(f"✓ Time tensor: {t}")
                    
                    # Test velocity function
                    velocity = policy._velocity_wrapper(t, x)
                    print(f"✓ Velocity function output shape: {velocity.shape}")
                    print(f"✓ Expected shape: {x.shape}")
                    
                    if velocity.shape == x.shape:
                        print("✓ Velocity function works correctly!")
                        return True
                    else:
                        print(f"✗ Shape mismatch: got {velocity.shape}, expected {x.shape}")
                        return False
                        
                except Exception as e:
                    if any(keyword in str(e).lower() for keyword in ["dataset", "zarr", "file", "path"]):
                        print(f"✓ Workspace failed as expected due to missing dataset")
                        
                        # Try to create policy directly
                        print("Trying to create policy directly...")
                        policy_cfg = cfg.policy
                        policy = hydra.utils.instantiate(policy_cfg, device=torch.device('cpu'))
                        print(f"✓ Policy created directly: {type(policy).__name__}")
                        
                        # Test velocity function
                        batch_size = 1
                        action_dim = 2
                        horizon = 16
                        obs_dim = 55332
                        
                        x = torch.randn(batch_size, action_dim * horizon + obs_dim)
                        t = torch.tensor([0.5])
                        
                        print(f"✓ Input tensor shape: {x.shape}")
                        
                        velocity = policy._velocity_wrapper(t, x)
                        print(f"✓ Velocity function output shape: {velocity.shape}")
                        
                        if velocity.shape == x.shape:
                            print("✓ Velocity function works correctly!")
                            return True
                        else:
                            print(f"✗ Shape mismatch: got {velocity.shape}, expected {x.shape}")
                            return False
                    else:
                        print(f"✗ Unexpected error: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("=" * 50)
    print("VELOCITY FUNCTION TEST")
    print("=" * 50)
    
    success = test_velocity_function()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Velocity function test passed!")
    else:
        print("❌ Velocity function test failed.")
    print("=" * 50)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())