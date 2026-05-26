#!/usr/bin/env python3
"""Dimension sweep runner — E1(EMA), E2(Temp), E3(Denoising)"""
import os, sys, shutil, subprocess
from datetime import datetime

ROOT = "/home/bio/lhz/spatialGDC"
PYTHON = "/home/bio/miniconda3/envs/spCLUE/bin/python"
RUNNER = os.path.join(ROOT, "_run_single_sweep.py")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

BACKUP = f"dimension_sweep_backup_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(BACKUP, exist_ok=True)
for fn in ["spCLUE/spCLUE.py", "spCLUE/loss.py", "spCLUE/network.py"]:
    shutil.copy2(fn, os.path.join(BACKUP, os.path.basename(fn)))
print(f"Backed up to {BACKUP}")

def restore_backup():
    for fn in ["spCLUE.py", "loss.py", "network.py"]:
        src = os.path.join(BACKUP, fn)
        dst = os.path.join("spCLUE", fn)
        if os.path.exists(src):
            shutil.copy2(src, dst)

def run_exp(label, suffix):
    """Launch subprocess to run the experiment"""
    print(f"\n{'#'*60}\n# {label}\n{'#'*60}")
    result = subprocess.run(
        [PYTHON, RUNNER, label, suffix],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': ROOT}
    )
    stdout = result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
    print(stdout)
    if result.returncode != 0:
        print(f"ERROR (rc={result.returncode}):")
        print(result.stderr[-1000:])
        return False
    return True

# ==================== MAIN ====================
print("\n" + "="*60)
print("DIMENSION SWEEP — spatialGDC")
print("="*60)

# --- Baseline ---
print("\n[0] BASELINE")
run_exp("Baseline", "dimension_sweep_baseline")

# --- E1: EMA ---
print("\n[1] E1: EMA")
spclue_path = os.path.join(ROOT, "spCLUE/spCLUE.py")
with open(spclue_path) as f:
    code = f.read()

old_opt = (
    "        self.optimizer = torch.optim.Adam(\n"
    "            filter(lambda p: p.requires_grad, self.model.parameters()),\n"
    "            lr=self.learning_rate,\n"
    "            weight_decay=self.weight_decay,\n"
    "        )\n"
    "        max_ari = 0.3 if self.n_spot <= 10000 else 1.1"
)
new_opt = (
    "        self.optimizer = torch.optim.Adam(\n"
    "            filter(lambda p: p.requires_grad, self.model.parameters()),\n"
    "            lr=self.learning_rate,\n"
    "            weight_decay=self.weight_decay,\n"
    "        )\n"
    "        import copy\n"
    "        self.ema_model = copy.deepcopy(self.model)\n"
    "        self.ema_model.eval()\n"
    "        for p in self.ema_model.parameters():\n"
    "            p.requires_grad = False\n"
    "        max_ari = 0.3 if self.n_spot <= 10000 else 1.1"
)
code = code.replace(old_opt, new_opt)

old_step = "            self.optimizer.step()"
new_step = (
    "            self.optimizer.step()\n"
    "\n"
    "            with torch.no_grad():\n"
    "                for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):\n"
    "                    ema_p.data.mul_(0.999).add_(p.data, alpha=0.001)"
)
code = code.replace(old_step, new_step, 1)

with open(spclue_path, 'w') as f:
    f.write(code)

run_exp("EMA", "dimension_sweep_ema")
restore_backup()

# --- E2: Temperature 0.5 ---
print("\n[2a] E2: Temperature = 0.5")
loss_path = os.path.join(ROOT, "spCLUE/loss.py")
with open(loss_path) as f:
    lcode = f.read()
lcode = lcode.replace(
    "def __init__(self, temperature=0.2",
    "def __init__(self, temperature=0.5"
)
with open(loss_path, 'w') as f:
    f.write(lcode)

run_exp("Temp0.5", "dimension_sweep_temp_0_5")
restore_backup()

# --- E2: Temperature 0.7 ---
print("\n[2b] E2: Temperature = 0.7")
with open(loss_path) as f:
    lcode = f.read()
lcode = lcode.replace(
    "def __init__(self, temperature=0.2",
    "def __init__(self, temperature=0.7"
)
with open(loss_path, 'w') as f:
    f.write(lcode)

run_exp("Temp0.7", "dimension_sweep_temp_0_7")
restore_backup()

# --- E3: Denoising Reconstruction ---
print("\n[3] E3: Denoising Recon")
net_path = os.path.join(ROOT, "spCLUE/network.py")
with open(net_path) as f:
    ncode = f.read()

old_recon = "        x_Rec = self.relu(z_fuse @ self.Transform2.W.data.T) @ self.Transform1.W.data.T"
new_recon = "        z_fuse_noisy = z_fuse + 0.05 * torch.randn_like(z_fuse)\n        x_Rec = self.relu(z_fuse_noisy @ self.Transform2.W.data.T) @ self.Transform1.W.data.T"
ncode = ncode.replace(old_recon, new_recon)

with open(net_path, 'w') as f:
    f.write(ncode)

run_exp("Denoising", "dimension_sweep_denoising")
restore_backup()

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("SWEEP COMPLETE — Summary")
print("="*60)

import pandas as pd
for csv_name, label in [
    ("dimension_sweep_baseline.csv", "Baseline"),
    ("dimension_sweep_ema.csv", "EMA"),
    ("dimension_sweep_temp_0_5.csv", "Temp tau=0.5"),
    ("dimension_sweep_temp_0_7.csv", "Temp tau=0.7"),
    ("dimension_sweep_denoising.csv", "Denoising"),
]:
    path = os.path.join(ROOT, csv_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        valid = df[df["ARI"] != "Error"]
        if not valid.empty:
            print(f"  {label}: ARI={valid['ARI'].mean():.4f}  NMI={valid['NMI'].mean():.4f}")
        else:
            print(f"  {label}: (all errors)")
    else:
        print(f"  {label}: MISSING")

print("\nBackups: " + BACKUP)
