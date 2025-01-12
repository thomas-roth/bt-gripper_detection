import os
import cv2
import torch
from tqdm import tqdm

from detectron2.config.config import get_cfg
from detectron2.data.build import build_detection_test_loader
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.datasets.irl_kitchen_gripper_detection import register_all_irl_kitchen_gripper_detection
from detectron2.engine.defaults import default_argument_parser, default_setup
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from projects.GripperDetection.build_trajs import build_trajectory


TRAINED_MODEL_PATH_CAM_1 = "/home/temp_store/troth/outputs/gripper_detection/models/2025_01_05-19_47_58/model_final.pth"
TRAINED_MODEL_PATH_CAM_2 = "/home/temp_store/troth/outputs/gripper_detection/models/2025_01_12-13_57_08/model_final.pth"
NUM_SEQUNECES = 242


# TODO: look at other trajs of cam_1: good enough or need to train again?
# TODO: experiment with RDP_TOLERANCE (0.05 too low?)
# TODO: eval all sequences
# TODO: write script to generate & fill folder structure for VLM


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
    cfg.merge_from_file("/home/i53/student/troth/code/bt/detectron2/projects/GripperDetection/configs/gripper_detection.yaml")
    cfg.ROBOT_STATE_ANNOTATIONS_PATH = "/home/temp_store/troth/data/irl_kitchen_gripper_detection/robot_state_annotations/"
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


def eval_sequence(cfg, model, sequence: str, save_bboxes=True, save_trajs=True, only_build_first_traj=False):
    test_data_loader = build_detection_test_loader(cfg, sequence, collate_fn=lambda x: x[0]) # collate_fn to "unwrap" batch (batch_size=1)

    torch.cuda.empty_cache()
    with torch.no_grad():
        outputs = model(test_data_loader)

    seq_name = str.join('_', str(test_data_loader.dataset.__getitem__(0)['image_id']).split('_')[:-4])

    if save_bboxes:
        os.makedirs(cfg.OUTPUT_DIR + f"/eval/{seq_name}/bboxes", exist_ok=True)

        for i, (input, output) in tqdm(enumerate(zip(test_data_loader, outputs)), total=len(test_data_loader), desc="Visualizing images"):
            if len(output["instances"].pred_boxes.tensor) == 0:
                continue # skip frame if no gripper detected

            output["instances"] = _merge_bbox_instances(output["instances"])

            input_img = cv2.imread(input["file_name"])
            visualizer = Visualizer(input_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
            input_img_with_bboxes = visualizer.draw_instance_predictions(output["instances"].to("cpu"))
            cv2.imwrite(cfg.OUTPUT_DIR + f"/eval/{seq_name}/bboxes/{input['image_id']}_bbox_{i:02d}.jpg", input_img_with_bboxes.get_image()[:, :, ::-1])
    
    if save_trajs:
        os.makedirs(cfg.OUTPUT_DIR + f"/eval/{seq_name}/trajs", exist_ok=True)
        total_no_gripper_found_counter = 0

        for i, input in tqdm(enumerate(test_data_loader), total=1 if only_build_first_traj else len(test_data_loader), desc="Building trajectories"):
            input_img = cv2.imread(input["file_name"])
            anno_file_path = cfg.ROBOT_STATE_ANNOTATIONS_PATH + f"/{seq_name}.pickle" # remove cam & img nr from image_id
            img_with_trajectory, no_gripper_found_counter = build_trajectory(input_img, outputs, anno_file_path=anno_file_path, start_index=i)
            if i == 0:
                total_no_gripper_found_counter = no_gripper_found_counter

            cv2.imwrite(cfg.OUTPUT_DIR + f"/eval/{seq_name}/trajs/{input['image_id']}_traj_{i:02d}.jpg", img_with_trajectory)
            
            if only_build_first_traj:
                break
        
        if total_no_gripper_found_counter > 0:
            print(f"Warning: no gripper detected in {1 if only_build_first_traj else total_no_gripper_found_counter}/{1 if only_build_first_traj else len(test_data_loader)} frames in sequence {seq_name}")


def main(args, save_bboxes=True, save_trajs=True, only_build_first_traj=False, sequence=""):
    cfg = setup_cfg(args)

    model_cam_1 = build_model(cfg)
    DetectionCheckpointer(model_cam_1).load(TRAINED_MODEL_PATH_CAM_1)
    model_cam_1.eval()

    model_cam_2 = build_model(cfg)
    DetectionCheckpointer(model_cam_2).load(TRAINED_MODEL_PATH_CAM_2)
    model_cam_2.eval()

    register_all_irl_kitchen_gripper_detection()

    if sequence == "":
        # eval all sequences
        for i in tqdm(range(NUM_SEQUNECES), total=NUM_SEQUNECES, desc="Evaluating sequences"):
            eval_sequence(cfg, model_cam_1, f"irl_kitchen_gripper_detection_cam_1_seq_{i:03d}", save_bboxes=save_bboxes,
                          save_trajs=save_trajs, only_build_first_traj=only_build_first_traj)
            eval_sequence(cfg, model_cam_2, f"irl_kitchen_gripper_detection_cam_2_seq_{i:03d}", save_bboxes=save_bboxes,
                          save_trajs=save_trajs, only_build_first_traj=only_build_first_traj)
    else:
        eval_sequence(cfg, model_cam_1, sequence=sequence, save_bboxes=save_bboxes,
                      save_trajs=save_trajs, only_build_first_traj=only_build_first_traj)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args, save_bboxes=False, save_trajs=True, only_build_first_traj=False, sequence="irl_kitchen_gripper_detection_cam_1_seq_047")
