import argparse
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.general import load_model as load_asr_model, get_model_class
from l2augment.modelling import load_model as load_rl_models
from l2augment.data import load_dataloader
from l2augment.rollout import cpu_rollout


def load_asr_model_fn(config, vocab_size, model_class, state_dict):
    asr_model = load_asr_model(config, vocab_size, model_class)
    asr_model.load_state_dict(state_dict)
    return asr_model

def main(config):
    tokenizer = load_tokenizer()
    asr_model_class = get_model_class(config = config)
    asr_model_state_dict = None #TODO
    partial_load_asr_model_fn = partial(load_asr_model_fn, config, tokenizer.vocab_size(), asr_model_class, asr_model_state_dict)
    policy_net, value_net = load_rl_models(config)
    dataloader = load_dataloader(config)

    epochs = config.get("training", {}).get("epochs", 1)
    for epoch in range(epochs):
        for batch in dataloader:
            utts, refs = batch["utts"], batch["refs"]
            cpu_rollout(
                policy = policy_net,
                load_asr_model_fn = partial_load_asr_model_fn,
                tokenizer = tokenizer
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)




