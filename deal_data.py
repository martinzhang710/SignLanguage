"""
@ Create on 01-05-2022

@ author: PENG ZHANG
"""

import os
import cv2

labels = "./image/train/labels.txt"
# labels = "./image/test/labels.txt"

with open(labels, "w") as f:
    for i in range(0, 10):
        data_path = f"./image/test/{i}"
        for index, image_name in enumerate(os.listdir(data_path)):
            image_path = os.path.join(data_path, image_name)
            f.write(f"{image_path} {i}\n")


