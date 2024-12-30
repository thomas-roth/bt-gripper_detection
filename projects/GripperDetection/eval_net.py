import cv2
import torch
from tqdm import tqdm

from detectron2.config.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.datasets.irl_kitchen_gripper_detection import register_all_irl_kitchen_gripper_detection
from detectron2.engine.defaults import DefaultTrainer, default_argument_parser, default_setup
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer


model_path = "/home/temp_store/troth/outputs/gripper_detection/model_final.pth"


# wrapper to squeeze items from dataset to fit model input shape
class GripperDetectionDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file("projects/GripperDetection/configs/gripper_detection.yaml")
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def _merge_instances(instances):
    # merge instances of gripper bboxes to biggest bbox

    x_mins = instances.pred_boxes.tensor[:, 0]
    y_mins = instances.pred_boxes.tensor[:, 1]
    x_maxs = instances.pred_boxes.tensor[:, 2]
    y_maxs = instances.pred_boxes.tensor[:, 3]

    instances.pred_boxes.tensor = torch.stack([x_mins.min(), y_mins.min(), x_maxs.max(), y_maxs.max()]).unsqueeze(0)

    instances.pred_classes = torch.tensor([0]) # only one class

    # calculate new score as weighted average of scores of merged bboxes (weight is size percentage of merged bbox)
    bbox_weights = (x_maxs - x_mins) * (y_maxs - y_mins) / ((x_maxs.max() - x_mins.min()) * (y_maxs.max() - y_mins.min()))
    instances.scores = (bbox_weights * instances.scores).sum().unsqueeze(0) / bbox_weights.sum()

    return instances


def main(args):
    cfg = setup_cfg(args)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(model_path)

    model.eval()

    register_all_irl_kitchen_gripper_detection()

    test_data_loader = DefaultTrainer.build_test_loader(cfg, "irl_kitchen_gripper_detection_test")
    test_data_loader = GripperDetectionDatasetWrapper(test_data_loader.dataset)

    outputs = model(test_data_loader)

    i = 0
    for input, output in tqdm(zip(test_data_loader, outputs), total=len(test_data_loader), desc="Building images"):
        output["instances"] = _merge_instances(output["instances"])

        input_img = cv2.imread(input["file_name"])
        visualizer = Visualizer(input_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        input_img_with_bboxes = visualizer.draw_instance_predictions(output["instances"].to("cpu"))
        cv2.imwrite(cfg.OUTPUT_DIR + "/eval/" + f"output_kitchen_trained_{i}.jpg", input_img_with_bboxes.get_image()[:, :, ::-1])

        i += 1


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
