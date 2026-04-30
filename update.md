# Update

Launched the UFMR sequential eval heartbeat job on GPU 3 in a detached tmux session.

Command:
- `./exp/launch_scripts/run_eval_mimas_all.sh 3`

tmux session:
- `ufmr-gpu3`

Log file:
- `exp/results/UFMR_mimas/logs/ufmr-gpu3.log`

Current run:
- `mseloss/multiepoch/chime6.yaml`

Fixes already in place:
- repo installed editable in `flash_attn_pytorch2` with `pip install -e .`
- `run_eval_mimas.sh` exports `PYTHONPATH` to the repo root

Outputs will land under:
- `exp/results/UFMR_mimas/<variant>/<regime>/<dataset>.txt`
