import datetime
import os
from pathlib import Path
import pickle
import shutil
import cv2
import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm
from termcolor import colored

from detectron2.config.config import get_cfg
from detectron2.data.build import build_detection_test_loader
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.datasets.gripper_detection_calvin import register_calvin_datamodule
from detectron2.engine.defaults import default_setup
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from projects.GripperDetection_calvin.utils.build_trajs import build_trajectory
from projects.GripperDetection_calvin.utils.build_qwen2vl_dataset import build_dataset_entry, build_output_message, save_dataset
from projects.GripperDetection_calvin.utils.gen_traj_gifs import gen_traj_gifs


TRAINED_MODEL_PATH_CAM_1 = "/home/temp_store/troth/outputs/gripper_detection_calvin/models/2025_01_13-16_58_35/model_final.pth" #FIXME
TRAINED_MODEL_PATH_CAM_2 = "/home/temp_store/troth/outputs/gripper_detection_calvin/models/2025_01_12-13_57_08/model_final.pth" #FIXME
ROBOT_STATE_ANNOTATIONS_PATH = "/home/temp_store/troth/data/irl_kitchen_gripper_detection_calvin/robot_state_annotations/" #FIXME?
NUM_SEQUENCES_ALL = 242 #FIXME


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file(f"{str(Path(__file__).absolute().parent)}/configs/gripper_detection.yaml")

    cfg.SAVE_BBOXES = True #False
    cfg.SAVE_TRAJS = True #False
    cfg.HIDE_PAST_TRAJ = True
    cfg.FILTER_NO_GRIPPER_DETECTED = True
    cfg.BUILD_DATASET = False
    cfg.SEQUENCE = "gripper_detection_calvin_validation" #None

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def _merge_bbox_instances(instances):
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


def _visualize_and_save_bboxes(cfg, env, test_data_loader, outputs):
    seq_name = str.join('_', str(test_data_loader.dataset.__getitem__(0)['image_id']).split('_')[:-4])
    os.makedirs(cfg.OUTPUT_DIR + f"/eval/{seq_name}/bboxes", exist_ok=True)

    for i, (batched_input, output) in tqdm(enumerate(zip(test_data_loader, outputs)), total=len(test_data_loader),
                                            desc="Visualizing images"):
        if len(output["instances"].pred_boxes.tensor) == 0:
            continue # skip frame if no gripper detected

        input = batched_input[0] # only one image in batch

        output["instances"] = _merge_bbox_instances(output["instances"])

        input_img = cv2.imread(input["file_name"])
        visualizer = Visualizer(input_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        input_img_with_bboxes = visualizer.draw_instance_predictions(output["instances"].to("cpu"))

        cv2.imwrite(cfg.OUTPUT_DIR + f"/eval/{seq_name}/bboxes/{input['image_id']}_bbox_{i:02d}.jpg", input_img_with_bboxes.get_image()[:, :, ::-1])


def _build_and_save_trajs(cfg, test_data_loader, outputs, hide_past_traj, filter_no_gripper_detected=True):
    seq_name = str.join('_', str(test_data_loader.dataset.__getitem__(0)['image_id']).split('_')[:-4])
    cam_id = 1 if "cam_1" in next(iter(test_data_loader))[0]["image_id"] else 2

    os.makedirs(cfg.OUTPUT_DIR + f"/eval/{seq_name}/trajs", exist_ok=True)
    
    for i, batched_input in tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Building trajectories"):
        input = batched_input[0] # only one image in batch

        input_img = cv2.imread(input["file_name"])
        anno_file_path = ROBOT_STATE_ANNOTATIONS_PATH + f"/{seq_name}.pickle"
        if hide_past_traj:
            img_with_trajectory, _, no_gripper_detected = build_trajectory(input_img, outputs, anno_file_path, start_index=i)
        else:
            img_with_trajectory, _, no_gripper_detected = build_trajectory(input_img, outputs, anno_file_path)

        if filter_no_gripper_detected and no_gripper_detected:
            # filter out sequences where gripper detection rate too low
            print(colored(f"Did not save trajectories for cam_{cam_id} of \"{seq_name}\"", "yellow"))
            break

        cv2.imwrite(cfg.OUTPUT_DIR + f"/eval/{seq_name}/trajs/{input['image_id']}_traj_{i:02d}.jpg", img_with_trajectory)


def _build_dataset(cfg, test_data_loader, outputs, dataset_for_seq, curr_datetime: str):
    assert type(dataset_for_seq) == list
    
    first_input = next(iter(test_data_loader))[0] # only one image in batch

    seq_name = str.join('_', str(first_input["image_id"]).split('_')[:-4])
    cam_id = 1 if "cam_1" in first_input["image_id"] else 2

    input_img = cv2.imread(first_input["file_name"])
    anno_file_path = ROBOT_STATE_ANNOTATIONS_PATH + f"/{seq_name}.pickle"
    _, gripper_keypoints, no_gripper_detected = build_trajectory(input_img, outputs, anno_file_path)
    if no_gripper_detected:
        print(colored(f"Did not build dataset entries for cam_{cam_id} of \"{seq_name}\"", "yellow"))
        return dataset_for_seq

    # copy image to dataset dir & save path
    dest_path_dir = cfg.OUTPUT_DIR + f"/dataset/{curr_datetime}/images/cam_{cam_id}"
    os.makedirs(dest_path_dir, exist_ok=True)
    dest_path_img = os.path.join(dest_path_dir, f"{seq_name}.jpg")
    shutil.copy(first_input["file_name"], dest_path_img)
    dataset_entry_rel_image_path = f"../images/cam_{cam_id}/{seq_name}.jpg"

    # build output message content
    dataset_entry_output_message = build_output_message(gripper_keypoints["traj"], input_img.shape[0], input_img.shape[1], gripper_keypoints["open"],
                                                        gripper_keypoints["close"])
    
    # get prompt
    prompt = pickle.load(open(anno_file_path, "rb"))["language_description"][1] # always three versions, second is most detailed

    dataset_for_seq.append(build_dataset_entry(dataset_entry_rel_image_path, prompt, dataset_entry_output_message))

    return dataset_for_seq


def eval_sequence(cfg_gd, model, env, sequence: str, dataset=None, curr_datetime=None):
    test_data_loader = build_detection_test_loader(cfg_gd, sequence)

    outputs = []
    with torch.no_grad():
        for input in test_data_loader:
            output = model(input)[0]
            outputs.append(output)
            
            torch.cuda.empty_cache()
    
    if cfg_gd.SAVE_BBOXES:
        _visualize_and_save_bboxes(cfg_gd, env, test_data_loader, outputs)

    if cfg_gd.SAVE_TRAJS:
        _build_and_save_trajs(cfg_gd, test_data_loader, outputs)
    
    if cfg_gd.BUILD_DATASET:
        return _build_dataset(cfg_gd, test_data_loader, outputs, dataset, curr_datetime)


@hydra.main(config_path="calvin_conf", config_name="calvin")
def main(cfg_calvin: DictConfig):
    cfg_gd = setup_cfg(args=None)

    datamodule = hydra.utils.instantiate(cfg_calvin.datamodule)
    register_calvin_datamodule(datamodule)

    if cfg_gd.SEQUENCE == None:
        # eval all sequences

        models = []
        if cfg_gd.BUILD_DATASET:
            dataset = []
            curr_datetime = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

        for cam_id in [1, 2]:
            model = build_model(cfg_gd)
            DetectionCheckpointer(model).load(TRAINED_MODEL_PATH_CAM_1 if cam_id == 1 else TRAINED_MODEL_PATH_CAM_2)
            model.eval()
            models.append(model)

            for i in tqdm(range(NUM_SEQUENCES_ALL), total=NUM_SEQUENCES_ALL, desc=f"Building dataset for cam_{cam_id}" if cfg_gd.BUILD_DATASET
                          else f"Evaluating sequences for cam_{cam_id}"):
                
                curr_sequence = f"gripper_detection_calvin_cam_{cam_id}_seq_{i:03d}"
                env = hydra.utils.instantiate(cfg_calvin, curr_sequence)
                
                if cfg_gd.BUILD_DATASET:
                    dataset = eval_sequence(cfg_gd, models[cam_id-1], env, curr_sequence, dataset, curr_datetime)
                else:
                    eval_sequence(cfg_gd, models[cam_id-1], env, curr_sequence)
        
        if cfg_gd.SAVE_TRAJS:
            gen_traj_gifs()
        
        if cfg_gd.BUILD_DATASET:
            save_dataset(cfg_gd.OUTPUT_DIR, dataset, curr_datetime)
    else:
        # eval single sequence

        assert type(cfg_gd.SEQUENCE) == str

        if cfg_gd.BUILD_DATASET:
            dataset = []
            curr_datetime = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

        model = build_model(cfg_gd)
        DetectionCheckpointer(model).load(TRAINED_MODEL_PATH_CAM_1 if "cam_1" in cfg_gd.SEQUENCE else TRAINED_MODEL_PATH_CAM_2)
        model.eval()

        env = hydra.utils.instantiate(cfg_calvin, cfg_gd.SEQUENCE)

        if cfg_gd.BUILD_DATASET:
            dataset = eval_sequence(cfg_gd, model, env, dataset, curr_datetime)
            save_dataset(cfg_gd.OUTPUT_DIR, dataset, curr_datetime)
        else:
            eval_sequence(cfg_gd, model, env)


if __name__ == "__main__":
    main()
