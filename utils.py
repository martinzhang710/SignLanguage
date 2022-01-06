"""
@ Create on 01-05-2022 

@ author: PENG ZHANG
"""
import numpy as np
import cv2


def get_mean_std(root):
    image_path, label_path = [], []
    with open(root, "r") as f:
        file_names = f.readlines()
        for file_name in file_names:
            image_name = file_name.split()[0]
            label_name = int(file_name.split()[1])
            image_path.append(image_name)
            label_path.append([label_name])
    nimages, mean, std = 0, 0, 0
    for image_name in image_path:
        nimages += 1
        image = cv2.imread(image_name)
        # image = image / 255
        image = image.reshape(-1, 3)
        mean += image.mean(0)
        std += image.std(0)
    mean /= nimages
    std /= nimages
    return mean, std


if __name__ == "__main__":
    label_path = "./image/test/labels.txt"
    mean, std = get_mean_std(label_path)
    print(mean)  # [0.50645013 0.52172516 0.53503299]
    print(std)  # [0.09817318 0.09692749 0.06366135]

