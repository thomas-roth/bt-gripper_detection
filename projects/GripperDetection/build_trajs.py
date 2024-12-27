import cv2


def get_end_effector_center_points(object_detector, rollout_imgs: list):
    end_effector_center_points = []

    for img in rollout_imgs:
        # infer gripper bbox
        outputs = object_detector(img)
        bbox = outputs["instances"].pred_boxes[0]

        bbox_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        end_effector_center_points.append(bbox_center)
    
    return end_effector_center_points


def draw_trajectory(img, end_effector_center_points: list):
    for i in range(0, len(end_effector_center_points)-1):
        color = (0, 0, (i+1) // len(end_effector_center_points) * 255) # BGR
        cv2.line(img, end_effector_center_points[i], end_effector_center_points[i+1],
                 color=color, thickness=2)

    return img


def draw_interaction_markers(img):
    # TODO
    return img


def build_trajectory(object_detector, rollout_imgs: list):
    end_effector_center_points = get_end_effector_center_points(object_detector, rollout_imgs)
    img_with_trajectory = draw_trajectory(rollout_imgs[0], end_effector_center_points)
    img_with_interaction_markers = draw_interaction_markers(img_with_trajectory)

    return img_with_interaction_markers


def main(object_detector):
    rollouts = []

    for i, rollout_imgs in enumerate(rollouts):
        img_with_trajectory = build_trajectory(object_detector, rollout_imgs)
        cv2.imwrite(f"./rollout_{i}.jpeg", img_with_trajectory)
