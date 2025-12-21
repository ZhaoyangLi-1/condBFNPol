# condBFNPol

## Diffusion Policy (PushT) – Monitoring-friendly runner

We ship a self-contained PushT diffusion trainer modeled after the original Diffusion Policy setup. It logs train/val losses every epoch to CSV/JSONL and can push metrics to Weights & Biases.

### Train
```bash
python scripts/train_diffusion_pusht.py \
  --dataset data/pusht/pusht_cchi_v7_replay.zarr \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 8 \
  --checkpoint-every 50 \
  --eval-every 10 --eval-episodes 3 --eval-max-steps 300 \
  --wandb-mode online \
  --wandb-project dp-pusht \
  --wandb-entity <your_entity> \
  --wandb-name pusht-diffusion-run

Or load defaults from a config file:
```bash
python scripts/train_diffusion_pusht.py --config-file config/train_diffusion_pusht.yaml
```
```
Flags:
- `--output-dir` to override the default `results/pusht_diffusion/<timestamp>/`
- `--wandb-mode` can be `disabled|online|offline` (defaults to `disabled`)
- `--checkpoint-every` controls checkpoint cadence

### Transformer backbone (low-dim)
- Added `networks/transformer_for_diffusion.TransformerForDiffusion`, matching the diffusion_policy low-dim transformer (8+ layers, causal attention optional, obs-as-cond support). You can point Hydra configs to this backbone via `_target_: networks.transformer_for_diffusion.TransformerForDiffusion`.

### Diffusion Transformer (PushT agent_pos)
Train the low-dim transformer variant (no vision encoder, uses agent_pos obs only):
```bash
python scripts/train_diffusion_transformer_pusht.py --config-file config/train_diffusion_transformer_pusht.yaml
```
Flags are similar to the image trainer (dataset, epochs, batch_size, eval_every, W&B, etc.).

### Outputs
- Run dir: `results/pusht_diffusion/<timestamp>/`
- Metrics: `train_log.csv`, `train_log.jsonl`
- Checkpoints: `ckpt_ep{E}.pt`, `ckpt_final.pt`
- W&B (optional): logs `train/loss`, `val/loss` per epoch
