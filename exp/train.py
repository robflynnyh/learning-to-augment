import argparse
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.general import load_model as load_asr_model, get_model_class
from l2augment.modelling import load_model as load_rl_models
from l2augment.data import load_dataloader

def main(config):
    tokenizer = load_tokenizer()
    asr_model = load_asr_model(config, tokenizer.vocab_size(), get_model_class(config = config))
    policy_net, value_net = load_rl_models(config)
    dataloader = load_dataloader(config, asr_model, policy_net)

    epochs = config.get("training", {}).get("epochs", 1)
    for epoch in range(epochs):
        for batch in dataloader:
            rewards, seeds = batch["rewards"], batch["seeds"]
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)




