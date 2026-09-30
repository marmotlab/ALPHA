# ALPHA

Multi-agent path finding (MAPF) with reinforcement learning: the ALPHA graph
information and graph network on the PRIMAL3 backbone (environment, rewards,
PIBT shielding, PPO).

## Installation

```bash
conda env create -f MAPF.yml
conda activate MAPF
```

## Training

```bash
python driver.py
```

Hyperparameters are in `alg_parameters.py`. Training logs to Weights & Biases by
default: set `ENTITY`, `EXPERIMENT_PROJECT` and `EXPERIMENT_NAME` in
`RecordingParameters`, or set `WANDB = False`. Checkpoints are saved to
`./models/<EXPERIMENT_PROJECT>/<EXPERIMENT_NAME><time>/`.

## Testing

The test set is in `32_32_house_0.2_0.3/`: 32x32 house maps with 50, 100, 150,
200, 250 and 300 agents, 200 cases each.

1. In `run_the_instances.py`, set `MODEL_PATH` to the trained model folder
   (the one that contains `net_checkpoint.pkl`).
2. In `alg_parameters.py`, set:
   ```python
   N_AGENTS = 50      # one of 50 / 100 / 150 / 200 / 250 / 300
   EPISODE_LEN = 512
   ```
3. Run:
   ```bash
   python run_the_instances.py
   ```

The script runs the 200 cases in parallel with Ray (`NUM_CPUS` in
`run_the_instances.py`) and prints the success rate, average steps and reach rate.
Repeat steps 2-3 for each agent count.

## Project structure

| file | content |
| --- | --- |
| `driver.py` | training entry point (PPO) |
| `runner.py` | rollout workers |
| `run_the_instances.py` | evaluation on the test set |
| `alg_parameters.py` | all hyperparameters |
| `mapf_gym.py`, `map_generator.py` | MAPF environment and map generation |
| `alpha_graph.py` | ALPHA graph observations (skeleton nodes and agent intentions) |
| `model.py`, `net.py`, `alpha_graph_net.py`, `transformer.py`, `dual_comms.py` | network and PPO update |
| `pibt_shielding.py`, `pibt/` | PIBT action shielding |
| `expert_guidance.py`, `lacam3/` | LaCAM3 expert planner |

## License

MIT, see `LICENSE.txt`. `lacam3/` keeps its own license (`lacam3/LICENCE.txt`).
