import os
from PIL import Image
import numpy as np
import cv2


class LoadFocalFolder:
    def __init__(self, root, type, frame_range=None, focal_range=None):
        self.root = root
        self.frames = [str(i) for i in range(frame_range[0], frame_range[1]+1)]  #  newvideo1: "images/007.png", 65 start frame
        # self.frames = [str(i).zfill(3) for i in range(frame_range[0], frame_range[1]+1)]  #  nonvideo3: "images/005.png"
        self.frame_range = frame_range
        self.type = type # focal or images
        self.focal_images_path = []
        self.focal_range = focal_range
        self.images_2d_path = []

        # 2D images setting
        for frame in self.frames:
            type_path = os.path.join(self.root, frame, 'images')
            image_path = os.path.join(type_path, '007.png')  #  newvideo1
            # image_path = os.path.join(type_path, '005.png')  #  nonvideo3
            self.images_2d_path.append(image_path)

        # focal images setting
        for frame in self.frames:
            type_path = os.path.join(root, frame, self.type)
            focal_images_name = [f'{i:03d}.png' for i in range(100)]  # '000', '001',,,
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

    def __getitem__(self, idx):
        path = self.images_2d_path[idx]
        img = cv2.imread(path)
        focal_plane = self.focal_images_path[idx]

        return img, focal_plane


if __name__ == '__main__':
    dataloader = LoadFocalFolder(root='D:/dataset/newvideo1', type='focal', frame_range=(1, 300), focal_range=(0, 100))
    print(np.array(dataloader.focal_images_path).shape)
    img = dataloader[0]
    print(img)
