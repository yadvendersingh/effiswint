import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from facenet_pytorch import MTCNN
import torch
import numpy as np

class RemoveFacialFeatures(ImageOnlyTransform):
    def __init__(self, choice, p=0.5):
        super(RemoveFacialFeatures, self).__init__(p=p)
        self.choice = choice

    def apply(self, image_array, **params):
        try:
            device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
            # create the MTCNN model, `keep_all=True` returns all the detected faces 
            mtcnn = MTCNN(keep_all=True, device=device)
            _, _, landmarks = mtcnn.detect(image_array, landmarks=True)
            if (landmarks is None) or (len(landmarks[0]) != 5):
                return image_array

            img_width, img_height, _ = image_array.shape
            if self.choice == 0:
                left_eye_x,left_eye_y = landmarks[0,0,0], landmarks[0,0,1]
                right_eye_x,right_eye_y = landmarks[0,1,0], landmarks[0,1,1]
                eye_width = int(0.25*img_width)  # Adjust this value as needed
                eye_height = int(0.15*img_height)
                y_ini, y_end = int(left_eye_y - eye_height / 2), int(left_eye_y + eye_height / 2)
                x_ini, x_end = int(left_eye_x - eye_width / 2), int(left_eye_x + eye_width / 2)
                image_array[
                    y_ini : y_end,
                    x_ini : x_end,
                ] = 0
                y_ini, y_end = int(right_eye_y - eye_height / 2), int(right_eye_y + eye_height / 2)
                x_ini, x_end = int(right_eye_x - eye_width / 2), int(right_eye_x + eye_width / 2)
                image_array[
                    y_ini : y_end,
                    x_ini : x_end,
                ] = 0
            elif self.choice == 1:
                mouth_height = int(0.15*img_height)
                # Exclude the left eye region
                left_mouth_x, left_mouth_y = landmarks[0,3,0], landmarks[0,3,1]
                right_mouth_x, right_mouth_y = landmarks[0,4,0], landmarks[0,4,1]
                y_ini, y_end = int(left_mouth_y - mouth_height / 2), int(left_mouth_y + mouth_height / 2)
                x_ini, x_end = int(left_mouth_x), int(right_mouth_x)
                image_array[
                    y_ini : y_end,
                    x_ini : x_end,
                ] = 0
            elif self.choice == 2:
                nose_height = int(0.25*img_height)  # Adjust this value as needed
                nose_width = int(0.15*img_width)

                # Exclude the left eye region
                nose_x, nose_y = landmarks[0,2,0], landmarks[0,2,1]
                y_ini, y_end = int(nose_y - nose_height / 1.5), int(nose_y + nose_height / 3)
                x_ini, x_end = int(nose_x - nose_width / 2), int(nose_x + nose_width / 2)
                image_array[
                    y_ini : y_end,
                    x_ini : x_end,
                ] = 0
            return image_array
        except:
            return image_array