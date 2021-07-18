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


class LoadFocalFolder:
    def __init__(self, root, type, img_size=640, stride=32, frame_range=(20, 90), focal_range=None):
        self.root = root
        self.frames = os.listdir(root)
        self.frame_range = frame_range
        self.type = type  # focal or images
        self.focal_images_path = []
        self.focal_range = focal_range
        self.images_2d_path = []
        self.img_size = img_size
        self.stride = stride

        # frame range
        if self.frame_range is None:
            pass
        else:
            self.frames = self.frames[self.frame_range[0]: self.frame_range[1] + 1]

        # 2D images setting
        for frame in self.frames:
            type_path = os.path.join(self.root, frame, 'images')
            image = os.listdir(type_path)[5]
            image_path = os.path.join(type_path, image)
            self.images_2d_path.append(image_path)

        # focal images setting
        for frame in self.frames:
            type_path = os.path.join(root, frame, self.type)
            focal_images_name = os.listdir(type_path)  # '000', '001',,,
            focal_planes = []

            for focal_image in focal_images_name:
                focal_image_path = os.path.join(type_path, focal_image)
                focal_planes.append(focal_image_path)  # 'D:/dataset/NonVideo3_tiny\\000\\focal\\020.png',,

            # focal range
            if self.focal_range is None:
                pass
            else:
                focal_planes = focal_planes[self.focal_range[0]: self.focal_range[1] + 1]

            self.focal_images_path.append(focal_planes)
            # self.images_path.append(os.path.join(root, frame, self.type))

    def __getitem__(self, idx):
        frame_idx = self.frames[idx]
        path = self.images_2d_path[idx]
        img = cv2.imread(path)
        focal_planes_resize = []

        focal_plane = self.focal_images_path[idx]
        for focal_image_path in focal_plane:
            focal_image = cv2.imread(focal_image_path)
            focal_image_resize = letterbox(focal_image, self.img_size, stride=self.stride)[0]
            focal_image_resize = focal_image_resize[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            focal_image_resize = np.ascontiguousarray(focal_image_resize)
            focal_planes_resize.append(focal_image_resize)

        return frame_idx, path, focal_planes_resize, img

        # 2D Images setting

    # def set_images_path(self):
    #     images_path = []
    #     for frame in self.frames:
    #         type_path = os.path.join(self.root, frame, 'images')
    #         image = os.listdir(type_path)[5]
    #         image_path = os.path.join(type_path, image)
    #         images_path.append(image_path)
    #     return images_path


if __name__ == '__main__':
    dataloader = LoadFocalFolder(root='D:/dataset/NonVideo3_tiny', type='focal', focal_range=(20, 50))
    print(np.array(dataloader.focal_images_path).shape)
    img = dataloader[0]
    print(img)
