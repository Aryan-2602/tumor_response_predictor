# scripts/download_data.py
import kagglehub

def download_breast_data():
    path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")
    print("Path to dataset files:", path)

if __name__ == "__main__":
    download_breast_data()
