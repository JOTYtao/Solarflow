import os
import torch
os.environ["HYDRA_FULL_ERROR"] = "1"
import dotenv
import hydra
from omegaconf import DictConfig
from api import utils
from api.experiment import train
dotenv.load_dotenv(override=True)
import sys
from pathlib import Path
torch.set_float32_matmul_precision("high")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = str(Path(__file__).parent.absolute())
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
torch.cuda.empty_cache()
@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: DictConfig):
    utils.extras(config)
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    return train(config)
if __name__ == "__main__":
    main()
