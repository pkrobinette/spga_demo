# DEMO: Self-Preserving Genetic Algorithms vs. Safe Reinforcement Learning in Discrete Action Spaces

## Organization
| File | Description |
|------|-------------|
| train_spga.py | Trains an SPGA agent |
| train_srl.py | Trains an SLR agent |
| rollout_spga.py | Demos the trained SPGA agent by rolling out the learned policy in an extended time environment (T=200 -> T=1000) |
| rollout_srl.py | Demos the trained SRL agent by rolling out the learned policy in an extended time environment (T=200 -> T=1000)|

## How to Run
1. `Train`: To train each agent, run the following from this directory:
```bash
python train_spga.py
python train_srl.py
```

2. `Test`: To demo agents, run the following from this directory:
```bash
python rollout_spga.py --render
python rollout_srl.py --render
```

3. `Compare`: To compare the rollouts, run the following from this directory:
```bash
python compare_rollouts.py
```


