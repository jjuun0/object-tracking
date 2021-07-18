import os
from PIL import Image
import numpy as np
import cv2


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
              stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class LoadImagesFolder:
    def __init__(self, root, type, img_size=640, stride=32):
        self.root = root
        self.frames = os.listdir(root)
        self.type = type  # focal or images
        self.images_path = []
        self.img_size = img_size
        self.stride = stride
        for frame in self.frames:
            frame_list = []
            type_path = os.path.join(root, frame, self.type)
            image = os.listdir(type_path)[5]
            image_path = os.path.join(type_path, image)
            # frame_list.append(image_path)
            self.images_path.append(image_path)
            # self.images_path.append(os.path.join(root, frame, self.type))

    def __getitem__(self, idx):
        path = self.images_path[idx]
        img = cv2.imread(path)
        # img_np = np.array(img)

        # Padded resize
        img_resize = letterbox(img, self.img_size, stride=self.stride)[0]

        # Convert
        img_resize = img_resize[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_resize = np.ascontiguousarray(img_resize)
        return path, img_resize, img



if __name__ == '__main__':
    dataloader = LoadImagesFolder(root='D:/dataset/NonVideo3_tiny', type='images')
    print(len(dataloader.images_path))
    img = dataloader[0]
    print(img)