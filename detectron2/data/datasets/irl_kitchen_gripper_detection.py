import os
import pickle
from PIL import Image
from typing import Dict, List
import numpy as np

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.file_io import PathManager


CLASS_NAMES = ["gripper"]


def deserialize_anno(robot_data_path: str): # TODO?: change to xml tree parsing
    with open(robot_data_path, 'rb') as file:
        return pickle.load(file)


# TODO: remove after testing
tmp_dir = "/home/temp_store/troth/data/kit_irl_real_kitchen/lang/mdt_annotations/04_04_2024-15_53_21_0_17_79_banana_from_right_stove_to_sink_62/signal_dict.pickle"
robot_data = deserialize_anno(tmp_dir)


def load_kitchen_instances(dirname: str, split: str):
    """
    Load kitchen detection annotations to Detectron2 format.
    
    Args:
        dirname: name of data directory. Contains subdirs "file_ids", "annotations", "images"
        split: data split. One of "train", "test"
    """

    # Get file ids (TODO: format = cam-nr_task-instruction_rollout-id_img-id?)
    with PathManager.open(os.path.join(dirname, "file_ids", split + ".txt")) as file:
        file_ids = np.loadtxt(file, dtype=str)

    data : List[Dict] = []

    for file_id in file_ids:
        img_file = os.path.join(dirname, "images", file_id + ".jpeg")
        anno_file = os.path.join(dirname, "annotations", file_id + ".pickle")

        annos = deserialize_anno(anno_file)

        img = Image.open(img_file)
        
        img_info = {
            "file_name": img_file, # full path
            "height": img.shape[0],
            "width": img.shape[1],
            "image_id": file_id,
            "annotations": [{
                "bbox": [], # TODO: get bbox from annos
                "bbox_mode": BoxMode.XYXY_ABS, # TODO: check if correct
                "category_id": CLASS_NAMES.index("gripper") # only one class # TODO: get cls from annos
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
