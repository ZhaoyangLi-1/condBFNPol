#!/usr/bin/env python3
"""Debug ODE integration shapes."""

import sys
import pathlib
import torch

# Add project root to path
project_root = pathlib.Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ode_debug():
    """Debug ODE shapes."""
    print("Debugging ODE integration...")
    
    try:
        from policies.streaming_flow_unet_hybrid_image_policy import StreamingFlowUnetHybridImagePolicy
        
        shape_meta = {
            'action': {'shape': [2]},
            'obs': {
                'agent_pos': {'shape': [2], 'type': 'low_dim'},
                'image': {'shape': [3, 96, 96], 'type': 'rgb'}
            }
        }
        
        horizon = 16
        n_obs_steps = 2
        n_action_steps = 8
        
        policy = StreamingFlowUnetHybridImagePolicy(
            shape_meta=shape_meta,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            sigma0=0.0,
            sigma1=0.1,
            num_integration_steps=3,  # Very small for debugging
            device='cpu'
        )
        
        dummy_obs = {
            'image': torch.randn(1, n_obs_steps, 3, 96, 96),
            'agent_pos': torch.randn(1, n_obs_steps, 2),
        }
        
        # Get encoded observation
        obs_features = policy.obs_encoder(dummy_obs)
        batch_size = obs_features.shape[0]
        
        print(f"Obs features shape: {obs_features.shape}")
        print(f"Action dim: {policy.action_dim}")
        print(f"Horizon: {policy.horizon}")
        print(f"Action part size: {policy.action_dim * policy.horizon}")
        
        # Sample initial noise
        action_noise = torch.randn(
            batch_size, policy.action_dim * policy.horizon,
            device=policy._device, dtype=obs_features.dtype
        ) * policy.sigma0
        
        print(f"Action noise shape: {action_noise.shape}")
        
        # Concatenate for ODE
        x0 = torch.cat([action_noise, obs_features], dim=1)
        print(f"Initial state x0 shape: {x0.shape}")
        
        # Integration time span
        t_span = torch.linspace(0.0, 1.0, policy.num_integration_steps + 1, device=policy._device)
        print(f"Time span shape: {t_span.shape}")
        print(f"Time span: {t_span}")
        
        # Test velocity function directly
        print("\n--- Testing velocity function ---")
        test_t = torch.tensor([0.5])
        velocity = policy._velocity_wrapper(test_t, x0)
        print(f"Velocity output shape: {velocity.shape}")
        print(f"Input shape: {x0.shape}")
        print(f"Shapes match: {velocity.shape == x0.shape}")
        
        # Now try ODE integration
        print("\n--- Testing ODE integration ---")
        with torch.no_grad():
            ode_result = policy.neural_ode(x0, t_span)
            print(f"ODE result type: {type(ode_result)}")
            
            # Handle different return types
            if isinstance(ode_result, tuple):
                print(f"ODE returned tuple with {len(ode_result)} elements")
                trajectory = ode_result[1]  # Usually the second element is the trajectory
                print(f"Trajectory type: {type(trajectory)}")
                print(f"ODE trajectory shape: {trajectory.shape}")
            else:
                trajectory = ode_result
                print(f"ODE trajectory shape: {trajectory.shape}")
            
            print(f"Expected shape: [{len(t_span)}, {batch_size}, {x0.shape[1]}]")
            
            # Extract final state
            final_state = trajectory[-1]
            print(f"Final state shape: {final_state.shape}")
            
            # Extract action part
            action_pred = final_state[:, :policy.action_dim * policy.horizon]
            print(f"Action prediction shape (before reshape): {action_pred.shape}")
            print(f"Elements in action pred: {action_pred.numel()}")
            print(f"Expected elements for reshape [1, 16, 2]: {1 * 16 * 2}")
            
            # Try reshape
            try:
                action_pred_reshaped = action_pred.reshape(batch_size, policy.horizon, policy.action_dim)
                print(f"✓ Reshape successful: {action_pred_reshaped.shape}")
                return True
            except Exception as reshape_error:
                print(f"✗ Reshape failed: {reshape_error}")
                return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the debug test."""
    print("=" * 60)
    print("ODE DEBUG TEST")
    print("=" * 60)
    
    success = test_ode_debug()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ODE debug test passed!")
    else:
        print("❌ ODE debug test failed.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())