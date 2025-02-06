import os
import numpy as np
from xml.etree import ElementTree
from typing import Dict, List

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.file_io import PathManager


CLASS_NAMES = ["gripper"]
NUM_SEQUNECES = 242
CAM_1_PATH = "/home/temp_store/troth/data/irl_kitchen_gripper_detection_calvin/cam_1"
CAM_2_PATH = "/home/temp_store/troth/data/irl_kitchen_gripper_detection_calvin/cam_2"
CALVIN_PATH = "/home/temp_store/troth/data/calvin_debug_dataset"


def load_calvin_split(eps_path: str, ep_start_id: int, ep_end_id: int):
    """
    Load calvin detection annotations to Detectron2 format.
    
    Args:
        eps_path: path to the directory containing the episodes
        ep_start_id: first episode id
        ep_end_id: last episode id
    """

    data =[]

    for ep_number in range(ep_start_id, ep_end_id + 1):
        ep_path = os.path.join(eps_path, f"episode_{ep_number}.npz")

        ep_data = np.load(ep_path)

        img_info = {
            "file_name": "",# full path
            "height": 0,
            "width": 0,
            "image_id": ep_number,
            "annotations": [{
                "bbox": [],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0, # only one class
            }],
        }

        data.append(img_info)

    return data



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
            }            

        data.append(img_info)

    return data


def register_calvin_gripper_detection(name: str, eps_path: str, split: str):
    ep_start_id, ep_end_id = np.load(f"{eps_path}/ep_start_end_ids.npy")[0]

    DatasetCatalog.register(name, lambda: load_calvin_split(eps_path, ep_start_id, ep_end_id))
    MetadataCatalog.get(name).set(
        thing_classes=list(CLASS_NAMES), dirname=eps_path, split=split
    )
    MetadataCatalog.get(name).evaluator_type = "gripper_detection_calvin"


def register_all_calvin_gripper_detection():
    file_ids = []
    subdirnames = []
    for cam_id in [1, 2]:
        for i in range(NUM_SEQUNECES):
            with PathManager.open(f"/home/temp_store/troth/data/irl_kitchen_gripper_detection_calvin/file_ids/cam_{cam_id}_seq_{i:03d}.txt") as file:
                file_ids.append(np.loadtxt(file, dtype=str))
                subdirnames.append(str.join("_", file_ids[-1][0].split("_")[:-4]))

    splits_cam_1 = [(f"gripper_detection_calvin_cam_1_seq_{i:03d}", file_ids[i], f"{CAM_1_PATH}/{subdirnames[i]}", f"cam_1_seq_{i:03d}") for i in range(NUM_SEQUNECES)]
    splits_cam_2 = [(f"gripper_detection_calvin_cam_2_seq_{i:03d}", file_ids[NUM_SEQUNECES + i], f"{CAM_2_PATH}/{subdirnames[NUM_SEQUNECES + i]}", f"cam_2_seq_{i:03d}") for i in range(NUM_SEQUNECES)]
    SPLITS = splits_cam_1 + splits_cam_2


    SPLITS = [("gripper_detection_calvin_training", f"{CALVIN_PATH}/training", "training"),
              ("gripper_detection_calvin_validation", f"{CALVIN_PATH}/validation", "validation")]

    for name, eps_path, split in SPLITS:
        register_calvin_gripper_detection(name, file_ids, eps_path, split)


def register_calvin_datamodule(datamodule):
    
    for split in ["trainng", "validation"]:
        name = f"gripper_detection_calvin_{split}"
        dirname = datamodule.training_dir if split == "training" else datamodule.val_dir
        dataloader = datamodule.train_dataloader if split == "training" else datamodule.val_dataloader

        DatasetCatalog.register(name, lambda: dataloader)
        MetadataCatalog.get(name).set(
            thing_classes=list(CLASS_NAMES), dirname=dirname, split=split
        )
        MetadataCatalog.get(name).evaluator_type = "gripper_detection_calvin"
