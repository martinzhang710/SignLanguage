"""
@ Create on 01-05-2022 

@ author: PENG ZHANG
"""

import random
import cv2
import numpy as np
from test import *
from torchvision.models.resnet import resnet18


def get_func():
    while True:
        num = random.randint(2, 10)
        add_num1, add_num2 = [], []
        for i in range(10):
            for j in range(10):
                if i+j == num:
                    add_num1.append(i)
                    add_num2.append(j)
        if len(add_num1) == 1:
            add_index = 0
        else:
            add_index = random.randint(0, len(add_num1)-1)
        result = add_num1[add_index] + add_num2[add_index]
        return f"{add_num1[add_index]} + {add_num2[add_index]} = ", result

def draw_func(image):
    text, result = get_func()
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, text, (90, 350), font, 3, (255, 255, 255), 3)
    return image, result

if __name__ == "__main__":
    net = torch.load("./net18.pt").cuda()
    net.eval()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    vid = cv2.VideoCapture(0)
    count = 0
    while True:
        image_bk = np.zeros((600, 1000, 3), dtype=np.uint8)
        image_bk, result = draw_func(image_bk)
        k = 0
        while True:
            count += 1
            return_value, image = vid.read()
            image = cv2.resize(image, (224, 224))
            image_bk[200:424, 620:844,:] = image
            if count % 50 == 0:
                num1 = gtest(net, image)
                if num1 == result:
                    image_an = cv2.putText(image_bk, "YOU ARE RIGHT", (50, 150), font, 1.5, (255, 255, 255), 2)
                    image_an = cv2.putText(image_an, str(num1), (520, 350), font, 3, (0, 255, 255), 3)
                    cv2.imshow("image", image_an)
                    cv2.waitKey(2000)
                    break
                else:
                    image_bk = cv2.putText(image_bk, "YOU ARE WRONG", (50, 150), font, 1.5, (255, 255, 255), 2)
                    image_bk = cv2.putText(image_bk, str(num1), (520, 350), font, 3, (0, 255, 255), 3)
                    cv2.imshow("image", image_bk)
                    cv2.waitKey(2000)
                    break
            cv2.imshow("image", image_bk)
            k = cv2.waitKey(100)
        if k == 27:
            break
