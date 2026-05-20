# ROB-60 UVQLM Oracle Follow-up

This directory contains the UVQLM oracle follow-up requested after the RMM/RFM
oracle sweeps. It uses `UnconditionalMaskGenerator` as the oracle proposal
policy, with local Mimas checkpoints:

- UMLM: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/UMLM/modelgpu.pt`
- BVAE: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt`
- ASR: `/store/store5/data/acp21rjf_checkpoints/l2augment/asr/step_105360.pt`

The first queued comparison uses the strongest completed RMM/RFM adaptation
hyperparameters, `lr=1e-5` and `single_step_lr=2e-1`, over search repeats
`1, 2, 3, 4, 5, 10, 20, 50`.

Expected result file:

- `exp/results/repro/oracle/UVQLM/tedlium_lr1e-5_searchlr2e-1.txt`

Queued wrapper:

- `exp/results/repro/oracle/jobs/uvqlm_lr1e-5_searchlr2e-1_gpu.sh`
