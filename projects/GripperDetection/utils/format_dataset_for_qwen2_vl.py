def build_message_content(traj_points: list, img_width, img_height, open_gripper_points: list, close_gripper_points: list):
    traj_points_and_actions = []
    for traj_point in traj_points:
        # normalize coordinates to [0, 1] (to enable different image sizes)
        normalized_x = float(traj_point[0]) / img_width
        normalized_y = float(traj_point[1]) / img_height

        traj_points_and_actions.append(f"({normalized_x}, {normalized_y})")

        if traj_point in open_gripper_points:
            traj_points_and_actions.append("<action>Open Gripper</action>")
        
        if traj_point in close_gripper_points:
            traj_points_and_actions.append("<action>Close Gripper</action>")

    return "<ans>[" + str.join(", ", traj_points_and_actions) + "]</ans>"


def build_dataset_entry(images: list, message: str):
    return {
            "messages": [{
                "content": message,
                "role": "system"
            }],
            "images": images
    }
