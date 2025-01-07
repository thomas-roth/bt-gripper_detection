import os
import shutil

from tqdm import tqdm


SOURCE = "/home/temp_store/troth/data/kit_irl_real_kitchen/lang/mdt_annotations"
DEST = "/home/temp_store/troth/data/irl_kitchen_gripper_detection"

NUM_SEQUNECES = 242

# TODO: remove old images with 2 long img numbers (starting from 04_04_2024-15_57_53_0_85_146_pot_from_left_stove_to_sink_61)
# TODO: why was cam_1_seq_105.txt not created? there are images? (should be 05_04_2024-10_53_59_0_187_248_pot_from_right_to_left_stove_61)

for root_seq, dirs_seq, files_seq in tqdm(os.walk(SOURCE), total=NUM_SEQUNECES, desc="Copying images"):
    dirs_seq.sort()
    for i, dir_seq in enumerate(dirs_seq):
        for root_cam, dirs_cam, files_cam in os.walk(os.path.join(root_seq, dir_seq)):
            if len(files_cam) == 0:
                 print(f"Skip {root_cam}, no annotations found {dirs_cam}")
                 continue
            
            # copy robot state annotations
            if files_cam[0].endswith(".pickle"):
                dest_robot_state_annotations = os.path.join(DEST, f"robot_state_annotations/{dir_seq}.pickle")
                shutil.copy2(os.path.join(root_cam, files_cam[0]), dest_robot_state_annotations)
            
            # copy images
            dirs_cam.sort()
            for dir_cam in dirs_cam:
                    with open(os.path.join(DEST, f"file_ids/{dir_cam}_seq_{i:03d}.txt"), "w") as file_ids_file:
                        for root_img, dirs_img, files_img in os.walk(os.path.join(root_cam, dir_cam)):
                            files_img.sort()
                            for file_img in files_img:
                                if file_img.endswith(".jpeg"):
                                    dest_img = os.path.join(DEST, f"{dir_cam}/{dir_seq}/images/{dir_seq}_{dir_cam}_img_{int(file_img.split('.')[0]):03d}.jpeg")
                                    os.makedirs(os.path.dirname(dest_img), exist_ok=True)
                                    shutil.copy2(os.path.join(root_img, file_img), dest_img)

                                    file_ids_file.write(f"{dir_seq}_{dir_cam}_img_{int(file_img.split('.')[0]):03d}\n")
