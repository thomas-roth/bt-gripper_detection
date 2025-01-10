import os
import numpy as np
from xml.etree import ElementTree
from typing import Dict, List

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.file_io import PathManager


CLASS_NAMES = ["gripper"]
NUM_SEQUNECES = 242
CAM_1_PATH = "/home/temp_store/troth/data/irl_kitchen_gripper_detection/cam_1"
CAM_2_PATH = "/home/temp_store/troth/data/irl_kitchen_gripper_detection/cam_2"


def load_kitchen_instances(file_ids: list, dirname: str):
    """
    Load kitchen detection annotations to Detectron2 format.
    
    Args:
        file_ids: list of file ids (timestamp + sequence task + sequence length)
        dirname: name of data directory. Contains subdirs "images", "annotations"
    """

    data : List[Dict] = []

    for file_id in file_ids:
        img_file = os.path.join(dirname, "images", file_id + ".jpeg")

        is_seq_annotated = os.path.isdir(os.path.join(dirname, "annotations")) 

        if is_seq_annotated:
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
        else:
            img_info = {
                "file_name": img_file, # full path
                "image_id": file_id,
                # TODO: height & width needed?
            }            

        data.append(img_info)

    return data


def register_irl_kitchen_gripper_detection(name: str, file_ids: list, dirname: str, split: str):
    DatasetCatalog.register(name, lambda: load_kitchen_instances(file_ids, dirname))
    MetadataCatalog.get(name).set(
        thing_classes=list(CLASS_NAMES), dirname=dirname, split=split
    )
    MetadataCatalog.get(name).evaluator_type = "irl_kitchen_gripper_detection"


def register_all_irl_kitchen_gripper_detection():
    file_ids = []
    subdirnames = []
    for cam_id in [1, 2]:
        for i in range(NUM_SEQUNECES):
            with PathManager.open(f"/home/temp_store/troth/data/irl_kitchen_gripper_detection/file_ids/cam_{cam_id}_seq_{i:03d}.txt") as file:
                file_ids.append(np.loadtxt(file, dtype=str))
                subdirnames.append(str.join("_", file_ids[-1][0].split("_")[:-4]))

    splits_cam_1 = [(f"irl_kitchen_gripper_detection_cam_1_seq_{i:03d}", file_ids[i], f"{CAM_1_PATH}/{subdirnames[i]}", f"cam_1_seq_{i:03d}") for i in range(NUM_SEQUNECES)]
    splits_cam_2 = [(f"irl_kitchen_gripper_detection_cam_2_seq_{i:03d}", file_ids[NUM_SEQUNECES + i], f"{CAM_2_PATH}/{subdirnames[NUM_SEQUNECES + i]}", f"cam_2_seq_{i:03d}") for i in range(NUM_SEQUNECES)]
    SPLITS = splits_cam_1 + splits_cam_2

    for name, file_ids, dirname, split in SPLITS:
        register_irl_kitchen_gripper_detection(name, file_ids, dirname, split)
