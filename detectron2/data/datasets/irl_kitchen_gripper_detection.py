import os
import numpy as np
from xml.etree import ElementTree
from typing import Dict, List

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.file_io import PathManager


CLASS_NAMES = ["gripper"]


def load_kitchen_instances(dirname: str, split: str):
    """
    Load kitchen detection annotations to Detectron2 format.
    
    Args:
        dirname: name of data directory. Contains subdirs "file_ids", "annotations", "images"
        split: data split. One of "train", "test"
    """

    # Get file ids (format = timestamp_task_instruction_seq_len_cam_nr_img_nr)
    with PathManager.open(os.path.join(dirname, "file_ids", split + ".txt")) as file:
        file_ids = np.loadtxt(file, dtype=str)

    data : List[Dict] = []

    for file_id in file_ids:
        img_file = os.path.join(dirname, "images", file_id + ".jpeg")
        anno_file = os.path.join(dirname, "annotations", file_id + ".xml")

        with PathManager.open(anno_file) as file:
            annos_tree = ElementTree.parse(file)
        
        img_info = {
            "file_name": img_file, # full path
            "height": int(annos_tree.find("./size/height").text),
            "width": int(annos_tree.find("./size/width").text),
            "image_id": file_id,
            "annotations": [{
                "bbox": [float(annos_tree.find("./object/bndbox").find(corner).text)
                         for corner in ["xmin", "ymin", "xmax", "ymax"]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0, # only one class
            }],
        }

        data.append(img_info)

    return data


def register_irl_kitchen_gripper_detection(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_kitchen_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=list(CLASS_NAMES), dirname=dirname, split=split
    )
    MetadataCatalog.get(name).evaluator_type = "irl_kitchen_gripper_detection"


def register_all_irl_kitchen_gripper_detection():
    SPLITS = [
        ("irl_kitchen_gripper_detection_train", "/home/temp_store/troth/data/irl_kitchen_gripper_detection/cam_1", "train"), # TODO?: extract cam number from dirname (to cfg / fct args)
        ("irl_kitchen_gripper_detection_test", "/home/temp_store/troth/data/irl_kitchen_gripper_detection/cam_1", "test"),
    ]

    for name, dirname, split in SPLITS:
        register_irl_kitchen_gripper_detection(name, dirname, split)
