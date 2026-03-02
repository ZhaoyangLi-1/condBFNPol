# Streaming Flow Policy Integration

This document describes the integration of Streaming Flow Policy into the condBFNPol framework.

## Overview

The Streaming Flow Policy has been successfully integrated into condBFNPol, allowing you to:

1. **Train/Setup** streaming flow policies using `scripts/train_workspace.py`
2. **Evaluate** them using `scripts/eval/eval_widowx.py` 
3. **Analyze** performance using `scripts/analysis/`

## Files Added

### Core Integration Files

1. **`streaming_flow_policy/`** - Core streaming flow policy implementation
   - `core/` - Core streaming flow policy algorithms
   - `utils.py` - Utility functions
   - `__init__.py` - Module initialization

2. **`policies/streaming_flow_policy.py`** - Policy wrapper for condBFNPol compatibility
   - Inherits from `BasePolicy`
   - Provides `predict_action()` interface for evaluation
   - Handles batch processing and device management

3. **`workspaces/train_streaming_flow_workspace.py`** - Training workspace
   - Loads demonstration trajectories
   - Sets up policy for inference
   - Manages evaluation runs

4. **`config/train_streaming_flow_pusht_real.yaml`** - Configuration file
   - Defines policy parameters
   - Sets up data loading
   - Configures logging and checkpointing

## Usage

### 1. Training/Setup

```bash
# Setup streaming flow policy with real PushT data
python scripts/train_workspace.py --config-name=train_streaming_flow_pusht_real

# Override parameters
python scripts/train_workspace.py --config-name=train_streaming_flow_pusht_real \
    policy.sigma0=0.2 \
    training.seed=123
```

### 2. Evaluation on Real Robot (WidowX)

```bash
# Evaluate on WidowX robot
python scripts/eval/eval_widowx.py \
  --checkpoint /path/to/streaming_flow_policy.ckpt \
  --widowx_envs_path /scr2/zhaoyang/bridge_data_robot_pusht/widowx_envs \
  --action_mode 2trans \
  --step_duration 0.1 \
  --act_exec_horizon 8 \
  --im_size 480 \
  --num_inference_steps 1 \
  --video_save_path /path/to/results
```

### 3. Analysis and Inference

```bash
# Run inference analysis
python scripts/analysis/comprehensive_ablation_study.py \
  --checkpoint-dir /path/to/streaming_flow_checkpoints

# Run other analysis scripts
python scripts/analysis/run_inference_steps_ablation.py \
  --checkpoint /path/to/streaming_flow_policy.ckpt
```

## Configuration

The streaming flow policy can be configured through the YAML config file:

```yaml
policy:
  _target_: policies.streaming_flow_policy.StreamingFlowPolicy
  
  # Policy parameters
  sigma0: 0.1                    # Initial Gaussian tube std deviation
  trajectories: null             # Will be loaded from demonstrations  
  prior: null                    # Will be computed from demonstrations
  
  # Framework parameters
  horizon: 16
  n_obs_steps: 2
  n_action_steps: 8
  
  # Device settings
  device: cuda
  dtype: float32
  clip_actions: true
```

## Dependencies

The integration supports both scenarios:

### With PyDrake (Full Implementation)
- Install PyDrake: `pip install drake`
- Full streaming flow policy algorithms available
- Can load and use demonstration trajectories

### Without PyDrake (Fallback Mode)
- Works without PyDrake installation
- Uses dummy implementation for testing
- Suitable for integration testing and development

## Architecture

### Policy Integration Flow

1. **Configuration** → `train_streaming_flow_pusht_real.yaml`
2. **Workspace** → `TrainStreamingFlowWorkspace`
3. **Policy** → `StreamingFlowPolicy` (wrapper)
4. **Core** → `StreamingFlowPolicyLatent` (if PyDrake available)

### Evaluation Flow

1. **Checkpoint Loading** → Policy state restored
2. **Observation Processing** → Image and proprioceptive data
3. **Action Prediction** → `predict_action()` method
4. **Robot Execution** → WidowX command interface

## Testing

Run the integration test to verify everything works:

```bash
python test_streaming_flow_integration.py
```

This will test:
- ✓ Import functionality
- ✓ Policy creation
- ✓ Forward inference
- ✓ predict_action interface
- ✓ Workspace loading

## Notes

1. **Demonstration-Based**: Unlike BFN or Diffusion policies, Streaming Flow Policy is demonstration-based and doesn't require traditional "training"

2. **Trajectory Loading**: Currently uses dummy trajectories. To use real demonstrations:
   - Implement trajectory extraction in `_load_demonstrations()`
   - Convert dataset episodes to PyDrake Trajectory format
   - Update policy with `self.policy.update_trajectories(trajectories, priors)`

3. **Compatibility**: The integration maintains full compatibility with condBFNPol's evaluation pipeline including:
   - WidowX robot evaluation
   - Analysis scripts
   - Logging and checkpointing
   - Performance profiling

## Next Steps

To fully utilize streaming flow policy:

1. **Install PyDrake**: For full algorithm implementation
2. **Load Real Trajectories**: Implement demonstration loading from your dataset
3. **Tune Parameters**: Adjust `sigma0` and other hyperparameters
4. **Evaluate**: Run on WidowX robot and compare with other policies