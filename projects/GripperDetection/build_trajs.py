import pickle
import cv2
import numpy as np
from rdp import rdp


THRESHOLD_CLOSE_GRIPPER = 0.05 # minimum width of gripper to be considered open
OFFSET_GRIPPER_WIDTH = 4 # annotations are desired gripper width, they are "faster" than the images => prepend list with nans
RDP_TOLERANCE = 0.05


def get_end_effector_center_points(rollout_imgs: list, start_index: int):
    end_effector_center_points = []
    no_gripper_found_counter = 0

    for i in range(start_index, len(rollout_imgs)):
        if len(rollout_imgs[i]["instances"].pred_boxes.tensor) == 0:
            no_gripper_found_counter += 1
            end_effector_center_points.append((-1, -1)) # tuple required for array trafo, integer type required for rdp, all other values are >= 0
            continue

        bbox = rollout_imgs[i]["instances"].pred_boxes[0].tensor.squeeze().cpu().numpy()

        bbox_center = round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)
        end_effector_center_points.append(bbox_center)

    return np.array(end_effector_center_points), no_gripper_found_counter


def _get_nearest_non_none_end_effector_center_point(end_effector_center_points: list, i_none: int):
    for i in range(1, len(end_effector_center_points)):
        i_before = i_none - i
        if i_before in range(len(end_effector_center_points)) and not np.array_equal(end_effector_center_points[i_before], np.array((-1, -1))):
            return end_effector_center_points[i_before]
        i_after = i_none + i
        if i_after in range(len(end_effector_center_points)) and not np.array_equal(end_effector_center_points[i_after], np.array((-1, -1))):
            return end_effector_center_points[i_after]
    
    return np.array((-1, -1))


def draw_trajectory(img, end_effector_center_points: list, anno_file_path: str, start_index: int):
    des_gripper_width = pickle.load(open(anno_file_path, "rb"))["des_gripper_width"]
    cur_gripper_width = OFFSET_GRIPPER_WIDTH * [float("nan")] + des_gripper_width[:-OFFSET_GRIPPER_WIDTH] # fill up with values that always evaluate to False
    cur_gripper_width = cur_gripper_width[start_index:] # only build trajectory starting at current frame
    assert len(cur_gripper_width) == len(end_effector_center_points)

    gripper_open = cur_gripper_width[0] >= THRESHOLD_CLOSE_GRIPPER

    # fill end effector center points for frames with no gripper detected
    for i in range(len(end_effector_center_points)):        
        if np.array_equal(end_effector_center_points[i], np.array((-1, -1))):
            end_effector_center_points[i] = _get_nearest_non_none_end_effector_center_point(end_effector_center_points, i_none=i)
            if np.array_equal(end_effector_center_points[i], np.array((-1, -1))):
                # no end effector center point found, skip drawing
                continue

    # simpilfy trajectory using Ramer-Douglas-Peucker algorithm
    mask = rdp(end_effector_center_points, epsilon=RDP_TOLERANCE, return_mask=True)
    end_effector_center_points = end_effector_center_points[mask]
    cur_gripper_width = np.array(cur_gripper_width)[mask]

    for i in range(len(end_effector_center_points) - 1):
        color = (0, 0, round((i+1) / len(end_effector_center_points) * 255)) # BGR
        cv2.line(img, end_effector_center_points[i], end_effector_center_points[i+1], color=color, thickness=2)
        
        if gripper_open and cur_gripper_width[i] < THRESHOLD_CLOSE_GRIPPER:
            # close gripper => draw green circle
            cv2.circle(img, end_effector_center_points[i], 5, color=(0, 255, 0), thickness=2) # BGR
            gripper_open = False
        elif not gripper_open and cur_gripper_width[i] >= THRESHOLD_CLOSE_GRIPPER:
            # open gripper => draw blue circle
            cv2.circle(img, end_effector_center_points[i], 5, color=(255, 0, 0), thickness=2) # BGR
            gripper_open = True

    return img


def build_trajectory(input_img, rollout_imgs: list, anno_file_path: str, start_index=0):
    end_effector_center_points, no_gripper_found_counter = get_end_effector_center_points(rollout_imgs, start_index)
    img_with_trajectory = draw_trajectory(input_img, end_effector_center_points, anno_file_path, start_index)

    return img_with_trajectory, no_gripper_found_counter
