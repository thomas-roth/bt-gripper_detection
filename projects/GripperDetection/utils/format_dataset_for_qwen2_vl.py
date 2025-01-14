import json


def build_output_message(traj_points: list, img_width, img_height, open_gripper_points: list, close_gripper_points: list):
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


def build_dataset_entry(image, prompt: str, output_message: str):
    return {
            "messages": [{
                "content": f"<image>In the image, please execute the command described in <prompt>{prompt}</prompt>. " +
                            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal. " +
                            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(0.25, 0.32), (0.32, 0.17), " +
                            "(0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>. The tuple denotes " +
                            "the x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action. " +
                            "The coordinates should be floats ranging between 0 and 1, indicating the relative locations of the points in the image.",
                "role": "user"
            },{
                "content": output_message,
                "role": "assistant"
            }],
            "images": image
    }


def save_dataset(general_output_dir, dataset, curr_datetime):
    with(open(f"{general_output_dir}/dataset/{curr_datetime}/dataset.json", "w")) as dataset_file:
        json.dump(dataset, dataset_file, indent=2)

    dataset_info = {
        "dataset_name": {
            "file_name": "dataset.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
                },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
                }
        }
    }

    with open(f"{general_output_dir}/dataset/{curr_datetime}/dataset_info.json", "w") as dataset_info_file:
        json.dump(dataset_info, dataset_info_file, indent=2)
