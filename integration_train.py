import yaml

from training.train import training


if __name__ == '__main__':
    with open("training/train_config.yaml", "r") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

    training(config)
