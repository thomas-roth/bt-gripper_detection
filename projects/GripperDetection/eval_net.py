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


def eval_sequence(cfg, model, sequence: str, save_bboxes=True, save_trajs=True, hide_past_traj=True):
    test_data_loader = build_detection_test_loader(cfg, sequence)

    seq_name = str.join('_', str(test_data_loader.dataset.__getitem__(0)['image_id']).split('_')[:-4])

    outputs = []
    with torch.no_grad():
        for input in test_data_loader:
            output = model(input)[0]
            outputs.append(output)

            torch.cuda.empty_cache()
    
    if save_bboxes:
        os.makedirs(cfg.OUTPUT_DIR + f"/eval/{seq_name}/bboxes", exist_ok=True)

        for i, (batched_input, output) in tqdm(enumerate(zip(test_data_loader, outputs)), total=len(test_data_loader), desc="Visualizing images"):
            if len(output["instances"].pred_boxes.tensor) == 0:
                continue # skip frame if no gripper detected

            input = batched_input[0] # only one image in batch

            output["instances"] = _merge_bbox_instances(output["instances"])

            input_img = cv2.imread(input["file_name"])
            visualizer = Visualizer(input_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
            input_img_with_bboxes = visualizer.draw_instance_predictions(output["instances"].to("cpu"))

            cv2.imwrite(cfg.OUTPUT_DIR + f"/eval/{seq_name}/bboxes/{input['image_id']}_bbox_{i:02d}.jpg", input_img_with_bboxes.get_image()[:, :, ::-1])
    
    if save_trajs:
        os.makedirs(cfg.OUTPUT_DIR + f"/eval/{seq_name}/trajs", exist_ok=True)
        total_no_gripper_found_counter = 0

        for i, batched_input in tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Building trajectories"):
            input = batched_input[0] # only one image in batch

            input_img = cv2.imread(input["file_name"])
            anno_file_path = cfg.ROBOT_STATE_ANNOTATIONS_PATH + f"/{seq_name}.pickle"
            if hide_past_traj:
                img_with_trajectory, no_gripper_found_counter = build_trajectory(input_img, outputs, anno_file_path, start_index=i)
            else:
                img_with_trajectory, no_gripper_found_counter = build_trajectory(input_img, outputs, anno_file_path)

            cv2.imwrite(cfg.OUTPUT_DIR + f"/eval/{seq_name}/trajs/{input['image_id']}_traj_{i:02d}.jpg", img_with_trajectory)

            if i == 0:
                total_no_gripper_found_counter = no_gripper_found_counter
        
        if total_no_gripper_found_counter > 0:
            print(f"Warning: no gripper detected in {total_no_gripper_found_counter}/{len(test_data_loader)} frames in sequence {seq_name}")


def main(args, save_bboxes=True, save_trajs=True, hide_past_traj=True, sequence=None):
    cfg = setup_cfg(args)

    register_all_irl_kitchen_gripper_detection()

    if sequence == None:
        # eval all sequences

        models = []
        for cam_id in [1, 2]:
            model = build_model(cfg)
            DetectionCheckpointer(model).load(TRAINED_MODEL_PATH_CAM_1 if cam_id == 1 else TRAINED_MODEL_PATH_CAM_2)
            model.eval()
            models.append(model)

            for i in tqdm(range(NUM_SEQUNECES), total=NUM_SEQUNECES, desc=f"Evaluating sequences for cam_{cam_id}"):
                curr_sequence = f"irl_kitchen_gripper_detection_cam_{cam_id}_seq_{i:03d}"
                eval_sequence(cfg, models[cam_id-1], curr_sequence, save_bboxes, save_trajs, hide_past_traj)
    else:
        # eval single sequence

        assert type(sequence) == str

        model = build_model(cfg)
        DetectionCheckpointer(model).load(TRAINED_MODEL_PATH_CAM_1 if "cam_1" in sequence else TRAINED_MODEL_PATH_CAM_2)
        model.eval()

        eval_sequence(cfg, model, sequence, save_bboxes, save_trajs, hide_past_traj)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args, save_bboxes=False, save_trajs=True, hide_past_traj=True)
    #main(args, save_bboxes=True, save_trajs=True, hide_past_traj=True, sequence="irl_kitchen_gripper_detection_cam_1_seq_000")
