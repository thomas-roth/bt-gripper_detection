import datetime
import os
from detectron2.engine.defaults import default_argument_parser, default_setup
from detectron2.config.config import get_cfg
from detectron2.data.datasets.irl_kitchen_gripper_detection import register_all_irl_kitchen_gripper_detection
from tools.train_net import Trainer
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file("projects/GripperDetection/configs/gripper_detection.yaml")

    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + f"/models/{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    os.makedirs(cfg.OUTPUT_DIR)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    cfg = setup_cfg(args)

    register_all_irl_kitchen_gripper_detection()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return Trainer.test(cfg, model)
    else:
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
