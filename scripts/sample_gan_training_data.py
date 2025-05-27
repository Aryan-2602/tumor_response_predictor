import os
import random
import shutil
from collections import defaultdict
from glob import glob

def sample_images(src_root, dst_root, target_per_class=5000, max_per_patient=50):
    class_dirs = {
        "0": os.path.join(dst_root, "non_idc"),
        "1": os.path.join(dst_root, "idc")
    }

    # Create destination dirs
    for path in class_dirs.values():
        os.makedirs(path, exist_ok=True)

    sampled_counts = {"0": 0, "1": 0}
    seen_images = set()

    # Get all patient dirs from both root and IDC_regular_ps50_idx5
    all_patient_paths = []
    for root_dir in [src_root, os.path.join(src_root, "IDC_regular_ps50_idx5")]:
        if not os.path.isdir(root_dir):
            continue
        for patient_id in os.listdir(root_dir):
            p_path = os.path.join(root_dir, patient_id)
            if os.path.isdir(p_path):
                all_patient_paths.append(p_path)

    random.shuffle(all_patient_paths)

    for patient_path in all_patient_paths:
        for class_label in ["0", "1"]:
            if sampled_counts[class_label] >= target_per_class:
                continue

            class_dir = os.path.join(patient_path, class_label)
            if not os.path.isdir(class_dir):
                continue

            all_images = glob(os.path.join(class_dir, "*.png"))
            random.shuffle(all_images)
            selected = 0

            for img_path in all_images:
                if sampled_counts[class_label] >= target_per_class:
                    break
                if img_path in seen_images:
                    continue

                dest_name = f"{class_label}_{sampled_counts[class_label]}.png"
                dest_path = os.path.join(class_dirs[class_label], dest_name)
                shutil.copy(img_path, dest_path)

                seen_images.add(img_path)
                sampled_counts[class_label] += 1
                selected += 1
                if selected >= max_per_patient:
                    break

        if all(sampled_counts[c] >= target_per_class for c in ["0", "1"]):
            break

    print(f"Sampled {sampled_counts['0']} non-IDC and {sampled_counts['1']} IDC images into {dst_root}")

if __name__ == "__main__":
    sample_images(
        src_root="data/breast_histopathology_images",
        dst_root="data/gan_train",
        target_per_class=5000,
        max_per_patient=50
    )
