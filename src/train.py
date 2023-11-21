"""
Created by Andrew Silva on 11/13/23
Copied from https://git.ecker.tech/mrq/ai-voice-cloning and https://github.com/neonbjb/DL-Art-School

"""
import argparse
import yaml

from trainer import Trainer


def train(config_path):
    cfg = parse(config_path, is_train=True)

    trainer = Trainer(config_path, cfg)
    trainer.do_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, help='Path to training configuration file.', default='../lich_king/train.yaml', nargs='+')
    args = parser.parse_args()
    cfg_path = args.yaml

    with open(cfg_path, 'r') as file:
        opt_config = yaml.safe_load(file)

    from utils import parse
    train(cfg_path)