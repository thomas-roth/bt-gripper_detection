from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import get_cfg
from detectron2.engine.defaults import default_argument_parser, default_setup
from tools.train_net import Trainer


def add_gripper_detection_config(cfg):
    cfg.merge_from_file("configs/gripper_detection.yaml")


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    add_gripper_detection_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    cfg = setup_cfg(args)

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
