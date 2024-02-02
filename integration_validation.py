import yaml

from validation.validation import model_eval

if __name__ == '__main__':
    with open("validation/validation_config.yaml", "r") as eval_conf_file:
        config = yaml.load(eval_conf_file, Loader=yaml.FullLoader)

    model_eval(**config)
