import os
from collections import defaultdict

def count_images_by_class(data_root):
    class_counts = defaultdict(int)
    patient_counts = defaultdict(lambda: defaultdict(int))  # patient_id → {class: count}
    total_patient_images = defaultdict(int)
    # Traverse both base and IDC_regular_ps50_idx5
    subdirs = [data_root]
    idx5_path = os.path.join(data_root, "IDC_regular_ps50_idx5")
    if os.path.isdir(idx5_path):
        subdirs.append(idx5_path)

    for subdir in subdirs:
        for patient_id in os.listdir(subdir):
            patient_path = os.path.join(subdir, patient_id)
            if not os.path.isdir(patient_path):
                continue

            for class_label in ["0", "1"]:
                class_path = os.path.join(patient_path, class_label)
                if os.path.isdir(class_path):
                    files = [f for f in os.listdir(class_path) if f.endswith(".png")]
                    count = len(files)
                    class_counts[class_label] += count
                    patient_counts[patient_id][class_label] += count
                    total_patient_images[patient_id] += count

    return class_counts, patient_counts, total_patient_images

def analyze(data_root):
    class_counts, patient_counts, total_patient_images = count_images_by_class(data_root)
    num_patients = len(patient_counts)
    avg_imgs_per_patient = sum(total_patient_images.values()) / num_patients
    avg_class_0 = sum(p["0"] for p in patient_counts.values()) / num_patients
    avg_class_1 = sum(p["1"] for p in patient_counts.values()) / num_patients
    print(f"Total patients: {num_patients}")
    print(f"Total class 0 images: {class_counts['0']}")
    print(f"Total class 1 images: {class_counts['1']}")
    print(f"Avg images per patient: {avg_imgs_per_patient:.2f}")
    print(f"  └── Avg class 0 images: {avg_class_0:.2f}")
    print(f"  └── Avg class 1 images: {avg_class_1:.2f}")

if __name__ == "__main__":
    data_dir = "data/breast_histopathology_images"
    analyze(data_dir)
