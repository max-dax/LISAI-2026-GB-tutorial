"""Train the Part 2 population network: loudest-in-tile NPE under the population prior.

Fresh training data is generated every batch from the band-local population generator
(gb_population.sample_tiles), so the effective simulation budget is steps x batch and
overfitting is not a concern. The default architecture is a rational-quadratic neural
spline flow; --arch maf reproduces Part 1's affine architecture for comparison.

Local (laptop) usage:
    uv run python train_confusion_flow.py --steps 6000 --threads 4

Cluster (H100) usage -- identical script, bigger budget (CUDA is auto-detected):
    python train_confusion_flow.py --steps 30000 --batch 1024 --bank-size 262144

Arguments
---------
--steps           Optimizer steps. Every step sees a freshly generated batch, so the
                  total simulation budget is steps x batch (default 6000).
--batch           Examples per step (default 512).
--lr              Peak Adam learning rate; cosine-annealed to lr/30 (default 3e-4).
--arch            "nsf" (rational-quadratic spline, default) or "maf" (Part 1's affine
                  architecture, for comparison).
--num-flow-steps  Number of (linear + autoregressive) flow steps (default 5).
--hidden-dim      Hidden width of each autoregressive block (default 128).
--num-bins        Spline bins per dimension, nsf only (default 8).
--bank-size       Precomputed background-waveform bank entries. Bigger = more waveform
                  diversity; 65536 (default) needs ~130 MB, 262144 ~540 MB.
--seed            Seed for bank, validation set, and training stream (default 0).
--threads         Torch CPU thread cap; 0 uses all cores (default 0).
--out             Checkpoint path (default checkpoints/part2_population_flow.pt).

The checkpoint stores the architecture, the label standardization, and the population
settings it was trained for, so the notebook can verify it evaluates the network on the
population it was trained on. To use a checkpoint trained elsewhere, copy the .pt file
into checkpoints/.
"""

import argparse
import time
from dataclasses import asdict

import numpy as np
import torch

import gb_population as gbp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000,
                    help="optimizer steps; each sees a fresh batch")
    ap.add_argument("--batch", type=int, default=512, help="examples per step")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="peak Adam learning rate (cosine-annealed to lr/30)")
    ap.add_argument("--arch", choices=["nsf", "maf"], default="nsf",
                    help="spline flow (nsf) or Part 1's affine flow (maf)")
    ap.add_argument("--num-flow-steps", type=int, default=5,
                    help="number of (linear + autoregressive) flow steps")
    ap.add_argument("--hidden-dim", type=int, default=128,
                    help="hidden width of each autoregressive block")
    ap.add_argument("--num-bins", type=int, default=8,
                    help="spline bins per dimension (nsf only)")
    ap.add_argument("--bank-size", type=int, default=65536,
                    help="background-waveform bank entries")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for bank, validation set, and training stream")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all)")
    ap.add_argument("--out", default="checkpoints/part2_population_flow.pt")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    device = gbp.device
    torch.manual_seed(args.seed)
    pop = gbp.PopulationSettings()

    print(f"device {device}, arch {args.arch}, {args.steps} steps x batch {args.batch} "
          f"= {args.steps * args.batch / 1e6:.1f}M fresh examples", flush=True)
    bank = gbp.build_bank(pop, n_bank=args.bank_size)
    stats = gbp.label_stats(bank, pop)

    d_va, th_va, _ = gbp.sample_tiles(bank, 4096, pop, seed=999)
    z_va = gbp.to_std(th_va, stats)

    arch = dict(arch=args.arch, num_flow_steps=args.num_flow_steps,
                hidden_dim=args.hidden_dim, num_bins=args.num_bins)
    flow = gbp.create_flow(**arch).to(device)
    print(f"{sum(p.numel() for p in flow.parameters()):,} trainable parameters")

    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr / 30)
    history = {"step": [], "train": [], "val": []}
    ema, t0 = None, time.time()

    def save():
        torch.save({"state_dict": flow.state_dict(), "arch": arch, "stats": stats.cpu(),
                    "population": asdict(pop), "history": history,
                    "meta": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                             "bank_size": args.bank_size, "seed": args.seed,
                             "seconds": time.time() - t0, "device": str(device)}},
                   args.out)

    for step in range(1, args.steps + 1):
        d, th, _ = gbp.sample_tiles(bank, args.batch, pop)
        loss = -flow.log_prob(gbp.to_std(th, stats), d).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
        sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 250 == 0 or step == args.steps:
            flow.eval()
            with torch.no_grad():
                val = float(np.mean([-flow.log_prob(z_va[i:i + 1024], d_va[i:i + 1024])
                                     .mean().item() for i in range(0, 4096, 1024)]))
            flow.train()
            history["step"].append(step)
            history["train"].append(ema)
            history["val"].append(val)
            rate = step / (time.time() - t0)
            print(f"step {step:>6}/{args.steps}  train (ema) {ema:7.3f}  val {val:7.3f}  "
                  f"[{rate:.2f} it/s, eta {(args.steps - step) / rate / 60:.0f} min]",
                  flush=True)
            save()          # crash-safe: the checkpoint tracks the last validation point

    save()
    print(f"saved {args.out} after {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
