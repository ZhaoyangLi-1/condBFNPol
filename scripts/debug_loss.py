"""Debug script to investigate small loss values.

Checks:
1. Normalized action statistics (Are they vanishing?).
2. BFN Sigma scaling impact.
"""

import torch
import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np


def get_sigma1(policy):
    """Robustly retrieves sigma_1 from any policy type."""
    # 1. Check bfn_config (ConditionalBFNPolicy / BFNPolicy)
    if hasattr(policy, "bfn_config") and hasattr(policy.bfn_config, "sigma_1"):
        return policy.bfn_config.sigma_1

    # 2. Check config attribute (BFNPolicy generic)
    if hasattr(policy, "config") and hasattr(policy.config, "sigma_1"):
        return policy.config.sigma_1

    # 3. Check guidance config (GuidedBFNPolicy doesn't use sigma_1 explicitly for loss scaling usually,
    #    but flow matching typically assumes sigma=0 or learns it.
    #    However, if it's BFN-style flow matching, it might have it.)

    # GuidedBFNPolicy uses Flow Matching (x1 - x0). It doesn't have sigma_1 in the same way
    # BFN does (which is variance of noise).
    # If this is GuidedBFNPolicy, return None or a dummy value.
    return None


@hydra.main(config_path="../config", config_name="config_guided", version_base=None)
def main(cfg):
    print(f"--- Debugging Loss for {cfg.task.policy._target_} ---")

    # 1. Load Data
    print("Instantiating DataModule...")
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup()
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))

    obs, actions = batch  # These are raw from datamodule

    # 2. Load Policy
    print("Instantiating Policy...")
    policy = instantiate(cfg.task.policy)
    policy.set_normalizer(datamodule.normalizer)

    # 3. Check Normalization
    print("\n--- Normalization Check ---")
    # Manually normalize to see values
    # Note: DataModule fits normalizer but yields raw data.
    # Policy normalizes internally. We emulate policy logic here.

    n_actions = policy.normalizer["action"].normalize(actions)

    print(
        f"Raw Actions:   Mean={actions.mean():.4f}, Std={actions.std():.4f}, Min={actions.min():.4f}, Max={actions.max():.4f}"
    )
    print(
        f"Norm Actions:  Mean={n_actions.mean():.4f}, Std={n_actions.std():.4f}, Min={n_actions.min():.4f}, Max={n_actions.max():.4f}"
    )

    if n_actions.std() < 0.1:
        print(
            "[WARNING] Normalized actions have very low variance! Loss will be artificially small."
        )
    elif n_actions.std() > 10:
        print("[WARNING] Normalized actions have huge variance! Loss will explode.")
    else:
        print("[OK] Normalized actions look healthy (Std approx 1).")

    # 4. Check BFN Parameters
    sigma_1 = get_sigma1(policy)
    print(f"\n--- BFN Config Check ---")

    if sigma_1 is not None:
        print(f"Sigma_1: {sigma_1}")
        # The loss scale factor is roughly 1/sigma^2 for the 'clean data' term in some parameterizations
        print(f"Expected Loss Scale Factor (approx 1/sigma^2): {1 / (sigma_1**2):.2f}")

        if sigma_1 > 0.1:
            print(
                "[WARNING] Sigma_1 is very large. This naturally shrinks the loss value but reduces precision."
            )
    else:
        print("Policy does not use Sigma_1 (likely Flow Matching or Diffusion).")

    # 5. Dry Run Loss
    print("\n--- Dry Run ---")
    # Move to device if possible (optional for dry run, but good practice)
    # policy.to("cpu")
    policy.train()
    with torch.no_grad():
        # Pass raw batch, policy handles norm
        # GuidedBFNPolicy expects tuple or dict
        loss = policy.compute_loss({"obs": obs, "action": actions})
        print(f"Computed Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
