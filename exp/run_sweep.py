from eval import main as run_eval
from omegaconf import OmegaConf
import argparse
import wandb
from functools import partial

def launch_sweep_agent(eval_config):
    wandb.init()
    sweep_config = wandb.config
    eval_config = OmegaConf.to_container(eval_config, resolve=True)

    assert 'evaluation' in eval_config, "Evaluation key must be present in config!"

    if 'lr' in sweep_config:
        if 'optim_args' not in eval_config['evaluation']:
            eval_config['evaluation']['optim_args'] = {}
        eval_config['evaluation']['optim_args']['lr'] = sweep_config['lr']

    if 'epochs' in sweep_config:
        eval_config['evaluation']['epochs'] = sweep_config['epochs']

    if 'augmentation_repeats' in sweep_config:
        if 'augmentation_config' not in eval_config['evaluation']:
            eval_config['evaluation']['augmentation_config'] = {'use_random': False}
        eval_config['evaluation']['augmentation_config']['repeats'] = sweep_config['augmentation_repeats']

    wer = run_eval(config = eval_config)
    wandb.log({'wer': wer})
    wandb.finish()


def main(eval_config, sweep_config, project_name, sweep_id=""):
    sweep_config = OmegaConf.to_container(sweep_config, resolve=True)
    sweep_id = wandb.sweep(sweep_config, project=project_name) if sweep_id == "" else sweep_id
    wandb.agent(sweep_id, function=partial(launch_sweep_agent, eval_config), project=project_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--sweep_config", "-sweep_config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--indexes', '-indexes', type=int, nargs='+', help='Indexes of the data to evaluate', default=[-1]) # -1 means all
    parser.add_argument('--project_name', '-project_name', type=str, help='Name of the project', default="l2_augment_sweeps")
    parser.add_argument('--sweep_id', type=str, help='Sweep ID to resume. If string is empty, a new sweep is created', default="")
    args = parser.parse_args()
    config = OmegaConf.load(args.eval_config)
    sweep_config = OmegaConf.load(args.sweep_config)
    config['indexes'] = args.indexes
    main(config, sweep_config, args.project_name, args.sweep_id)




