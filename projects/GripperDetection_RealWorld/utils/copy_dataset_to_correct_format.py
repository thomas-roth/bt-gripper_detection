import os
import shutil
from natsort import natsorted

from tqdm import tqdm


SOURCE = "/home/temp_store/troth/data/kit_irl_real_kitchen/lang/mdt_annotations"
DEST = "/home/temp_store/troth/data/irl_kitchen_gripper_detection_real_world"

NUM_SEQUNECES = 242


for root_seq, dirs_seq, _ in tqdm(os.walk(SOURCE), total=NUM_SEQUNECES, desc="Copying images"):
    dirs_seq = natsorted(dirs_seq)
    for i, dir_seq in enumerate(dirs_seq):
        for root_cam, dirs_cam, files_cam in os.walk(os.path.join(root_seq, dir_seq)):
            if len(files_cam) == 0:
                print(f"Skip {root_cam}, no annotations found, i = {i}")
                continue
            
            # copy robot state annotations
            if files_cam[0].endswith(".pickle"):
                dest_robot_state_annotations = os.path.join(DEST, f"robot_state_annotations/{dir_seq}.pickle")
                shutil.copy2(os.path.join(root_cam, files_cam[0]), dest_robot_state_annotations)
            
            # copy images
            dirs_cam = natsorted(dirs_cam)
            for dir_cam in dirs_cam:
                    with open(os.path.join(DEST, f"file_ids/{dir_cam}_seq_{i:03d}.txt"), "w") as file_ids_file:
                        for root_img, _, files_img in os.walk(os.path.join(root_cam, dir_cam)):
                            files_img = natsorted(files_img)
                            for file_img in files_img:
                                if file_img.endswith(".jpeg"):
                                    dest_img = os.path.join(DEST, f"{dir_cam}/{dir_seq}/images/{dir_seq}_{dir_cam}_img_{int(file_img.split('.')[0]):03d}.jpeg")
                                    os.makedirs(os.path.dirname(dest_img), exist_ok=True)
                                    shutil.copy2(os.path.join(root_img, file_img), dest_img)

                                    file_ids_file.write(f"{dir_seq}_{dir_cam}_img_{int(file_img.split('.')[0]):03d}\n")
