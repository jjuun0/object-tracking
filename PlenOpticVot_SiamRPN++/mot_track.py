import os
import cv2
import torch
from pathlib import Path, PurePosixPath
from datetime import datetime

from config_parser import ConfigParser
from get_frame import get_frames
from IOU import IOU
from model import ModelBuilder
from tracker import build_tracker

import sys
import numpy as np
import time
import math

import copy
import matplotlib.pyplot as plt


def save_image(save_dir, frame_num, frame):
    ''' output 이미지 저장 '''
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / '{:03d}.jpg'.format(frame_num)
    cv2.imwrite(str(save_path), frame)


def ground_truth(center, size):
    f = open("ground_truth/Video3.txt", 'a')
    data = "%d,%d,%d,%d\n" % (center[0], center[1], size[0], size[1])
    f.write(data)
    f.close()


def get_distance(d1, d2):
    return math.sqrt(((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2))


def draw_bboxes(img, tracker_num, bbox, color, identities=None, offset=(0, 0)):
    x, y, w, h = bbox
    label = str(tracker_num)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
    cv2.rectangle(img, (x, y), (x + t_size[0], y + t_size[1]), color, -1)
    cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def main():
    # config parsing & setting
    config = ConfigParser('./config.json')
    exper_name = config['name']
    is_gt_on = config['is_gt_on']
    is_record = config['is_record']
    video_name = config['video_name']
    video_type = config['video_type']
    img2d_ref = config['image2d_ref']
    start_focal_num = config['start_focal_num']
    last_focal_num = config['last_focal_num']
    ckpt_path = config['pretrained_model']
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    save_path = Path(config['save_path']) / timestamp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    # # ground truth
    # if is_gt_on:    # IoU 정확도를 측정할 것인지
    #     f = open('ground_truth/Non_video4_GT.txt', 'r')  # GT 파일

    print("Please type the number of trackers: ")
    tracker_num = int(sys.stdin.readline())
    print()

    # create model
    tracker = []
    for _ in range(tracker_num):
        model = ModelBuilder()
        # load model
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cpu())
        model.load_state_dict(ckpt)
        model.eval().to(device)
        build_tracker(model)
        tracker.append(build_tracker(model))

    # tracking
    is_first_frame = True
    frame_num = 0

    current_target = [-1 for _ in range(tracker_num)]
    bbox_color = [-1 for _ in range(tracker_num)]
    first_frame_center = [(-1, -1) for _ in range(tracker_num)]
    prior_frame_center = [(-1, -1) for _ in range(tracker_num)]
    current_frame_center = [(-1, -1) for _ in range(tracker_num)]
    prior_distance = [-1 for _ in range(tracker_num)]
    current_distance = [-1 for _ in range(tracker_num)]
    necessary_tracking = [True for _ in range(tracker_num)]
    pass_frame = [0 for _ in range(tracker_num)]
    first_time = [True for _ in range(tracker_num)]

    output = [-1 for _ in range(tracker_num)]
    init_rect = [(-1, -1, -1, -1) for _ in range(tracker_num)]
    max_bbox = [(-1, -1, -1, -1) for _ in range(tracker_num)]
    max_center_bbox = [(-1, -1) for _ in range(tracker_num)]

    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)

    for frame, focals in get_frames(video_name, video_type, img2d_ref, start_focal_num, last_focal_num):
        frame_start = time.time()
        if is_first_frame:
            try:
                for k in range(tracker_num):
                    init_rect[k] = cv2.selectROI(video_name, frame, True, False)
                    tracker[k].init(frame, init_rect[k])
                    bbox_color[k] = list(np.random.random(size=3) * 256)
                print()
            except:
                exit()

            is_first_frame = False
            total_start = time.time()
        else:
            frame_num += 1
            print(f'[{frame_num}] frame ')
            max_index = [-1 for _ in range(tracker_num)]
            max_val = [-1 for _ in range(tracker_num)]

            for k in range(tracker_num):
                if first_time[k] or pass_frame[k] > 1:
                    pass_frame[k] = 0
                # if first_time[k]:

                    start = time.time()

                    for i, f in enumerate(focals):
                        # image = copy.deepcopy(frame)
                        img = cv2.imread(f)
                        output[k] = tracker[k].track(img)

                        if output[k]['best_score'] > max_val[k]:
                            max_val[k] = output[k]['best_score']
                            max_index[k] = i
                            max_center_bbox[k] = (output[k]['cx'], output[k]['cy'])
                            max_bbox[k] = list(map(int, output[k]['bbox']))
                            # draw_bboxes(image, k + 1, max_bbox[k], color[k])
                            # cv2.imshow(video_name, image)
                            # outputs = output[k]['outputs']
                            # test = outputs['cls'][0].detach().cpu().permute(1, 2, 0).numpy()
                            # for i in range(10):  # anchor : 5, cls : 2 (positive, negative)
                            #     plt.subplot(2, 5, i+1)
                            #     plt.imshow(test[:, :, i])
                            #
                            # plt.show()

                    # update tracker
                    tracker[k].center_pos = np.array(max_center_bbox[k])
                    tracker[k].size = np.array(max_bbox[k][2:])
                    # draw_bboxes(image, k+1, max_bbox[k], color[k])
                    # cv2.imshow(video_name, image)

                    end = time.time()
                    print(f"[{k + 1}] entire tracking time : {end - start:.2f} sec")

                    first_frame_center[k] = max_center_bbox[k]
                    current_target[k] = max_index[k]

                    first_time[k] = False
                    # print('------------------------')
                    # print()
                else:
                    # for k in range(tracker_num):
                    if necessary_tracking[k]:
                        start = time.time()
                        range_focals = focals[current_target[k] - 3: current_target[k] + 4]
                        for i, f in enumerate(range_focals):
                            # image = copy.deepcopy(frame)
                            img = cv2.imread(f)
                            output[k] = tracker[k].track(img)

                            if output[k]['best_score'] > max_val[k]:
                                max_val[k] = output[k]['best_score']
                                max_index[k] = i + current_target[k] - 3
                                max_center_bbox[k] = (output[k]['cx'], output[k]['cy'])
                                max_bbox[k] = list(map(int, output[k]['bbox']))

                                # draw_bboxes(image, k + 1, max_bbox[k], color[k])
                                # cv2.imshow(video_name, image)
                                # outputs = output[k]['outputs']
                                # test = outputs['cls'][0].detach().cpu().permute(1, 2, 0).numpy()
                                # for i in range(10):  # anchor : 5, cls : 2 (positive, negative)
                                #     plt.subplot(2, 5, i + 1)
                                #     plt.imshow(test[:, :, i])
                                # plt.show()

                        tracker[k].center_pos = np.array(max_center_bbox[k])
                        tracker[k].size = np.array(max_bbox[k][2:])
                        # max_index_center[k] = (output[k]['cx'], output[k]['cy'])

                        if prior_frame_center[k] == (-1, -1):
                            prior_frame_center[k] = max_center_bbox[k]
                            prior_distance[k] = get_distance(prior_frame_center[k], first_frame_center[k])

                        else:
                            current_frame_center[k] = max_center_bbox[k]
                            current_distance[k] = get_distance(current_frame_center[k], prior_frame_center[k])

                            if current_distance[k] < prior_distance[k]:  # 속도 감소
                                necessary_tracking[k] = False

                            # else:  # 속도 증가
                            #     prior_frame_center[k] = current_frame_center[k]
                            #     prior_distance[k] = current_distance[k]
                            prior_frame_center[k] = current_frame_center[k]
                            prior_distance[k] = current_distance[k]

                        end = time.time()
                        print(f"[{k + 1}] update tracking time : {end - start:.2f} sec")

                        current_target[k] = max_index[k]

                    else:
                        necessary_tracking[k] = True
                        pass_frame[k] += 1
                        print(f'[{k + 1}] pass tracking frame')

            # print(current_target)

            for k in range(tracker_num):
                # print(tracker[k].center_pos, tracker[k].size)
                draw_bboxes(frame, k+1, max_bbox[k], bbox_color[k])
            cv2.imshow(video_name, frame)

            # save_path = os.path.join('D:/dataset/NonVideo3_tiny_result/SiamRPN++', '{:03d}.png'.format(frame_num))
            # cv2.imwrite(save_path, frame)

            # # ground truth
            # if is_gt_on:
            #     line = f.readline()
            #     bbox_label = line.split(',')
            #     bbox_label = list(map(int, bbox_label))
            #
            #     iou = IOU(bbox, bbox_label)
            #
            #     labelx = bbox_label[0] + (bbox_label[2] / 2)
            #     labely = bbox_label[1] + (bbox_label[3] / 2)
            #
            #     pre = ((outputs[max_index]['cx'] - labelx)**2 +
            #            (outputs[max_index]['cy'] - labely)**2) ** 0.5
            #
            #     if is_record:
            #         result_iou = open('ground_truth/result_iou.txt', 'a')
            #         result_iou.write(str(iou) + ',')
            #         result_iou.close()
            #
            #         result_pre = open('ground_truth/result_pre.txt', 'a')
            #         result_pre.write(str(pre) + ',')
            #         result_pre.close()
            #
            #     cv2.rectangle(frame, (bbox_label[0], bbox_label[1]),
            #                   (bbox_label[0]+bbox_label[2],
            #                    bbox_label[1]+bbox_label[3]),
            #                   (255, 255, 255), 3)
            if is_record:
                save_image(save_path, frame_num, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

        if frame_num:
            print(f'[{frame_num}] total tracking time : {time.time() - frame_start:.2f} sec')
            print()
            print()

    print(f'total tracking time : {time.time() - total_start:.2f} sec')
if __name__ == "__main__":
    main()
