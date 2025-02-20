import os

os.environ["HYDRA_FULL_ERROR"] = "1"
import dotenv
import hydra
from omegaconf import DictConfig
from core import utils
from experiments.train import train
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@hydra.main(version_base="1.3", config_path="E:/research/my_code/solar_flow/configs/", config_name="SwinLSTM.yaml")
def main(config: DictConfig):
    utils.extras(config)
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    return train(config)

if __name__ == "__main__":
    main()
