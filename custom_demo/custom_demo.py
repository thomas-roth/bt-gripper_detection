import cv2

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import Visualizer
from detectron2.data import MetadataCatalog


def setup_config():
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    return cfg


def main():
    setup_logger()
    cfg = setup_config()

    img = cv2.imread("./input_kitchen.jpeg")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)

    print("classes: ", outputs["instances"].pred_classes)
    print("boxes: ", outputs["instances"].pred_boxes)

    visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    img_with_bboxes = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("./output_kitchen.jpeg", img_with_bboxes.get_image()[:, :, ::-1])

if __name__ == "__main__":
    main()
