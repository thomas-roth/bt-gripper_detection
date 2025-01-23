import datetime
import os
import pickle
import shutil
import cv2
import torch
from tqdm import tqdm
from termcolor import colored

from detectron2.config.config import get_cfg
from detectron2.data.build import build_detection_test_loader
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.datasets.irl_kitchen_gripper_detection import register_all_irl_kitchen_gripper_detection
from detectron2.engine.defaults import default_argument_parser, default_setup
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from projects.GripperDetection.utils.build_trajs import build_trajectory
from projects.GripperDetection.utils.build_qwen2vl_dataset import build_dataset_entry, build_output_message, save_dataset
from projects.GripperDetection.utils.gen_traj_gifs import gen_traj_gifs


TRAINED_MODEL_PATH_CAM_1 = "/home/temp_store/troth/outputs/gripper_detection/models/2025_01_13-16_58_35/model_final.pth"
TRAINED_MODEL_PATH_CAM_2 = "/home/temp_store/troth/outputs/gripper_detection/models/2025_01_12-13_57_08/model_final.pth"
ROBOT_STATE_ANNOTATIONS_PATH = "/home/temp_store/troth/data/irl_kitchen_gripper_detection/robot_state_annotations/"
NUM_SEQUENCES_ALL = 242


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file("projects/GripperDetection/configs/gripper_detection.yaml")
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


def _visualize_and_save_bboxes(cfg, test_data_loader, outputs):
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


def eval_sequence(cfg, model, sequence: str, save_bboxes=False, save_trajs=False, hide_past_traj=True, filter_no_gripper_detected=True,
                  build_dataset=False, dataset=None, curr_datetime=None):
    test_data_loader = build_detection_test_loader(cfg, sequence)

    outputs = []
    with torch.no_grad():
        for input in test_data_loader:
            output = model(input)[0]
            outputs.append(output)
            
            torch.cuda.empty_cache()
    
    if save_bboxes:
        _visualize_and_save_bboxes(cfg, test_data_loader, outputs)

    if save_trajs:
        _build_and_save_trajs(cfg, test_data_loader, outputs, hide_past_traj, filter_no_gripper_detected)
    
    if build_dataset:
        return _build_dataset(cfg, test_data_loader, outputs, dataset, curr_datetime)


def main(args, save_bboxes=False, save_trajs=False, hide_past_traj=True, filter_no_gripper_detected=True, build_dataset=False, sequence=None):
    cfg = setup_cfg(args)

    register_all_irl_kitchen_gripper_detection()

    if sequence == None:
        # eval all sequences

        models = []
        if build_dataset:
            dataset = []
            curr_datetime = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

        for cam_id in [1, 2]:
            model = build_model(cfg)
            DetectionCheckpointer(model).load(TRAINED_MODEL_PATH_CAM_1 if cam_id == 1 else TRAINED_MODEL_PATH_CAM_2)
            model.eval()
            models.append(model)

            for i in tqdm(range(NUM_SEQUENCES_ALL), total=NUM_SEQUENCES_ALL, desc=f"Building dataset for cam_{cam_id}" if build_dataset
                          else f"Evaluating sequences for cam_{cam_id}"):
                
                curr_sequence = f"irl_kitchen_gripper_detection_cam_{cam_id}_seq_{i:03d}"
                
                if build_dataset:
                    dataset = eval_sequence(cfg, models[cam_id-1], curr_sequence, save_bboxes, save_trajs, hide_past_traj, filter_no_gripper_detected,
                                            build_dataset, dataset, curr_datetime)
                else:
                    eval_sequence(cfg, models[cam_id-1], curr_sequence, save_bboxes, save_trajs, hide_past_traj, filter_no_gripper_detected)
        
        if save_trajs:
            gen_traj_gifs()
        
        if build_dataset:
            save_dataset(cfg.OUTPUT_DIR, dataset, curr_datetime)
    else:
        # eval single sequence

        assert type(sequence) == str

        if build_dataset:
            dataset = []
            curr_datetime = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

        model = build_model(cfg)
        DetectionCheckpointer(model).load(TRAINED_MODEL_PATH_CAM_1 if "cam_1" in sequence else TRAINED_MODEL_PATH_CAM_2)
        model.eval()

        if build_dataset:
            dataset = eval_sequence(cfg, model, sequence, save_bboxes, save_trajs, hide_past_traj, build_dataset, dataset, curr_datetime)
            save_dataset(cfg.OUTPUT_DIR, dataset, curr_datetime)
        else:
            eval_sequence(cfg, model, sequence, save_bboxes, save_trajs, hide_past_traj)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    #main(args, save_bboxes=True, save_trajs=True, hide_past_traj=True, build_dataset=True, sequence="irl_kitchen_gripper_detection_cam_1_seq_050")
    #main(args, save_trajs=True, hide_past_traj=False, sequence="irl_kitchen_gripper_detection_cam_1_seq_042")
    #main(args, save_trajs=True, hide_past_traj=False)
    main(args, build_dataset=True)
