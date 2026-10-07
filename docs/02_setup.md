# 2. Setting up and running

Previous: [01_concepts.md](01_concepts.md). Next: [03_code_walkthrough.md](03_code_walkthrough.md).

---

## 2.1 What you need

* **Python 3.10 to 3.13.** PyTorch does not yet publish packages for newer versions. On
  Windows, [Anaconda](https://www.anaconda.com/download) is the easiest way to get one.
  Check with `python --version`.
* **About 1 GB of disk** for PyTorch.
* **A GPU is optional.** Everything runs on a CPU. Training is several times faster on an
  NVIDIA GPU, because the code records each SAC update as a CUDA graph (see 3.6).

## 2.2 Install

```bash
git clone <repository URL>
cd <the folder that was created>
pip install -r requirements.txt
```

If several Pythons are installed, make sure `pip` and `python` belong to the same one
(`python -m pip install -r requirements.txt` is the safe form).

## 2.3 First commands, in order

```bash
python tests/test_env.py          # 8 sanity checks, about 10 s; every line should say "ok"
python demo.py                    # the pretrained agent survives the combined fault (about 1 min)
python demo.py --pid              # the classical controller with the calibrator
python calibration_study.py       # the estimator on its own: slow vs fast vs triggered (about 1 min)
```

`demo.py` prints one line per flight and saves `results/demo/trajectory.png` and
`results/demo/episode.gif`. Open the GIF: the right-hand panels show the estimates
snapping to the new values when the change arrives.

## 2.4 Training

```bash
# a one-minute check that training works
python train.py --run-name smoke --total-steps 30000

# the self-calibrating agent (about 30 to 50 minutes on a laptop GPU)
python train.py --run-name my_calib

# the other agents in the comparison: change only what they observe, or what they train on
python train.py --run-name my_blind   --set observation.mode=base
python train.py --run-name my_history --set observation.mode=history
python train.py --run-name my_oracle  --set observation.mode=oracle
python train.py --run-name my_nominal --set observation.mode=base uncertainty.enabled=false
```

Every 50,000 steps training prints a line like:

```
step   500000 | train return   812.4 crash 0.06 | eval return   905.1 crash 0.02 success 0.92 final dist  0.08 m | alpha 0.010 |  14.2 min  (best)
```

* **train return / crash:** the last 200 training episodes, which include random
  exploration;
* **eval return / crash / success / final dist:** 48 held-out flights flown without
  randomness. This is the honest number. `(best)` means the model was saved as `best.pt`.

## 2.5 Evaluating

```bash
python evaluate.py --runs results/my_calib results/my_blind --pid --episodes 100
python plot_training.py --runs results/my_*
```

`evaluate.py` flies every controller through the six test scenarios on identical drones,
then writes `results/eval/summary.csv` and the curves used by the figures. With
`--episodes 100` and seven controllers it takes a few minutes.

## 2.6 How long things take

Measured on a laptop (Intel i5-13420H, RTX 3050 6 GB), with other work running:

| task | time |
|---|---|
| sanity checks | 10 s |
| `demo.py` with GIF | 1 min |
| `calibration_study.py` | 1 min |
| 1M training steps, alone on the GPU | about 20 to 30 min |
| several trainings in parallel | they share the GPU; about 80 % of its time goes to two runs |
| full evaluation (100 flights per scenario) | 3 to 6 min |

## 2.7 Common problems

| symptom | cause and fix |
|---|---|
| `No module named torch` | you are on a Python without PyTorch, often a newer one (3.14). Run `python --version` and use 3.10 to 3.13. |
| `PermissionError ... train_log.csv` | an older version of the logger met a OneDrive or Dropbox lock. The current logger keeps the file open; update the code, or train outside a synced folder. |
| training very slow on GPU | check `device cuda` in the first printed line, and that `sac.cuda_graph` is `auto`. Without the graph each update costs about 20 ms instead of 2 ms. |
| training slow on CPU | expected: about 30 to 40 ms per update. Lower `train.updates_per_step`, or train fewer steps. |
| `CUDA error ... graph` | rare driver problems with CUDA graphs: set `--set sac.cuda_graph=false`. The maths is identical. |
| results differ slightly from the README | GPU arithmetic is not bit-for-bit reproducible, so seeds give close but not identical runs. The README reports the spread over seeds for this reason. |
