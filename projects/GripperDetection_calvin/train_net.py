import datetime
import os

import hydra
from omegaconf import DictConfig
from detectron2.engine.defaults import default_setup
from detectron2.config.config import get_cfg
from detectron2.data.datasets.gripper_detection_calvin import register_calvin_datamodule
from tools.train_net import Trainer


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file("/home/i53/student/troth/code/bt/detectron2/projects/GripperDetection_calvin/configs/gripper_detection.yaml")

    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + f"/models/{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    os.makedirs(cfg.OUTPUT_DIR)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg

@hydra.main(config_path="calvin_conf", config_name="calvin")
def main(cfg_calvin: DictConfig) -> None:
    cfg_gd = setup_cfg(args=None)

    datamodule = hydra.utils.instantiate(cfg_calvin.datamodule)
    #datamodule.prepare_data() # TODO?: remove bc not needed bc shm false
    datamodule.setup()
    register_calvin_datamodule(datamodule)

    trainer = Trainer(cfg_gd)
    trainer.resume_or_load()
    return trainer.train()

if __name__ == "__main__":
    main()
