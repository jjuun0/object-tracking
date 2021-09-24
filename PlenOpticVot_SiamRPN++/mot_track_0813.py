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
from FocalDataloader import LoadFocalFolder
import niqe
import csv
import pandas as pd
from iou_jun import IoU


def save_image(save_dir, frame_num, frame):
    """ output 이미지 저장 """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / '{:03d}.jpg'.format(frame_num)
    cv2.imwrite(str(save_path), frame)


def ground_truth(center, size):
    f = open("ground_truth/Video3.txt", 'a')
    data = "%d,%d,%d,%d\n" % (center[0], center[1], size[0], size[1])
    f.write(data)
    f.close()


def get_distance(d1, d2):
    """ 거리 측정 """
    return math.sqrt(((d1[0] - d2[0]) ** 2) + ((d1[1] - d2[1]) ** 2))

def get_variance(array):
    return np.var(array)


def draw_bboxes(img, tracker_num, bbox, color, identities=None, offset=(0, 0)):
    x, y, w, h = bbox
    label = str(tracker_num)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    cv2.rectangle(img, (x, y), (x + t_size[0], y + t_size[1]), color, -1)
    cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def main():
    # config parsing & setting
    config = ConfigParser('./config.json')
    exper_name = config['name']
    is_gt_on = config['is_gt_on']
    write_gt = config['write_gt']
    is_record = config['is_record']
    video_name = config['video_name']
    video_type = config['video_type']
    img2d_ref = config['image2d_ref']
    start_frame_num = config['start_frame_num']
    last_frame_num = config['last_frame_num']
    start_focal_num = config['start_focal_num']
    last_focal_num = config['last_focal_num']
    ckpt_path = config['pretrained_model']
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    save_path = Path(config['save_path']) / timestamp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    print("Please type the number of trackers: ")
    tracker_num = int(sys.stdin.readline())
    print()

    # # ground truth
    if is_gt_on:    # IoU 정확도를 측정할 것인지
        gt = pd.read_csv('gt.csv')


    if write_gt:  # gt를 csv에 작성할것인지
        gt = open('gt.csv', 'w', newline='')
        wr = csv.writer(gt)
        wr.writerow(['x', 'y', 'h', 'w'] * tracker_num)



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
    # model = ModelBuilder()
    # # load model
    # ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cpu())
    # model.load_state_dict(ckpt)
    # model.eval().to(device)
    # for _ in range(tracker_num):
    #     build_tracker(model)
    #     tracker.append(build_tracker(model))

    # tracking
    is_first_frame = True
    frame_num = 0
    gt_index = 0

    current_target = [-1 for _ in range(tracker_num)]
    bbox_color = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]

    # 속도 측정을 위한 변수
    first_frame_center = [(-1, -1) for _ in range(tracker_num)]
    prior_frame_center = [(-1, -1) for _ in range(tracker_num)]
    current_frame_center = [(-1, -1) for _ in range(tracker_num)]
    prior_distance = [-1 for _ in range(tracker_num)]
    current_distance = [-1 for _ in range(tracker_num)]

    # 현재 프레임에 대해 트래킹을 스킵할지를 저장하는 변수
    necessary_tracking = [True for _ in range(tracker_num)]

    # 연속해서 프레임을 스킵했는지 저장하는 변수
    pass_frame = [0 for _ in range(tracker_num)]

    # 연속해서 Best Score 값이 낮게 나왔는지 저장하는 변수
    low_score = [0 for _ in range(tracker_num)]

    # current target 에서 +-d 포컬 영역을 볼지 판단하는 변수
    d_index = [0 for _ in range(tracker_num)]

    first_time = [True for _ in range(tracker_num)]
    output = [-1 for _ in range(tracker_num)]

    max_bbox = [(-1, -1, -1, -1) for _ in range(tracker_num)]
    max_center_bbox = [(-1, -1) for _ in range(tracker_num)]
    max_index = [-1 for _ in range(tracker_num)]
    max_output = [-1 for _ in range(tracker_num)]

    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
    frame_num += start_frame_num - 1

    for frame, focals in LoadFocalFolder(video_name, 'focal', frame_range=(start_frame_num, last_frame_num),
                                         focal_range=(start_focal_num, last_focal_num)):
        frame_num += 1

        frame_start = time.time()
        if is_first_frame:
            try:
                for k in range(tracker_num):
                    init_rect = cv2.selectROI(video_name, frame, True, False)
                    tracker[k].init(frame, init_rect)
                    # bbox_color[k] = list(np.random.random(size=3) * 256)
                print()
            except:
                exit()

            is_first_frame = False
            total_start = time.time()
        else:

            print(f'[{frame_num}] frame ')
            max_val = [-1 for _ in range(tracker_num)]

            for k in range(tracker_num):
                # [entire focal plane range tracking]
                # if first_time[k] or pass_frame[k] > 1 or low_score[k] > 1:
                if first_time[k] or low_score[k] > 2:
                    pass_frame[k] = 0
                    low_score[k] = 0

                    start = time.time()
                    # vgg_score = []

                    for i, f in enumerate(focals):
                        # image = copy.deepcopy(frame)
                        # print(f'[{i+start_focal_num} th focal index]')
                        img = cv2.imread(f)
                        # output[k] = tracker[k].track(img)
                        output[k] = tracker[k].track(img, frame_num, i+start_focal_num)
                        # vgg_score.append(output[k]['vgg_corr_sum'])

                        if output[k]['best_score'] > max_val[k]:
                            max_output[k] = output[k]
                            max_val[k] = output[k]['best_score']
                            max_index[k] = i

                    # vgg_score = sorted(vgg_score, key=lambda x: x[1])
                    # print(vgg_score)

                    max_center_bbox[k] = (max_output[k]['cx'], max_output[k]['cy'])
                    max_bbox[k] = list(map(int, max_output[k]['bbox']))

                    # update tracker
                    tracker[k].center_pos = np.array(max_center_bbox[k])
                    tracker[k].size = np.array(max_bbox[k][2:])

                    end = time.time()
                    print(f"[{k + 1}] entire focal plane range tracking time : {end - start:.2f} sec")

                    first_frame_center[k] = max_center_bbox[k]
                    prior_frame_center[k] = (-1, -1)
                    current_target[k] = max_index[k]

                    first_time[k] = False
                    # print()
                else:
                    # for k in range(tracker_num):
                    if necessary_tracking[k]:
                        start = time.time()

                        # [Focal plane range setting]

                        # 선명도 계산 결과 반영
                        if isinstance(current_target[k], list):
                            range_focals = []
                            for i in current_target[k]:
                                range_focals.append(focals[i])

                        # Focal plane range: current target - d ~ current_target + d
                        elif d_index[k] < current_target[k] < last_focal_num - d_index[k]:
                            range_focals = focals[current_target[k] - d_index[k]: current_target[k] + d_index[k] + 1]

                        # Focal plane range 범위 벗어나지 않게 처리
                        elif current_target[k] <= d_index[k]:
                            range_focals = focals[0: d_index[k] * 2 + 1]
                        else:
                            range_focals = focals[-(d_index[k] * 2 + 1):]
                        # print(f'[{k + 1}] current target : [{start_focal_num + current_target[k]}]')

                        for i, f in enumerate(range_focals):
                            # image = copy.deepcopy(frame)
                            focal_index = f[-7:-4]
                            print(f'[{focal_index} th focal index]')
                            img = cv2.imread(f)
                            output[k] = tracker[k].track(img, frame_num, focal_index)

                            if output[k]['best_score'] > max_val[k]:
                                # max_anchor_idx = output[k]['best_idx']
                                max_output[k] = output[k]
                                max_val[k] = output[k]['best_score']
                                if isinstance(current_target[k], list):
                                    max_index[k] = current_target[k][i]
                                else:
                                    max_index[k] = i + current_target[k] - d_index[k]

                        # Best Score 가 0.7 이상이면 기존의 방법 그대로 수행
                        if max_val[k] >= 0.7:
                            max_center_bbox[k] = (max_output[k]['cx'], max_output[k]['cy'])
                            max_bbox[k] = list(map(int, max_output[k]['bbox']))
                            current_target[k] = max_index[k]
                            tracker[k].center_pos = np.array(max_center_bbox[k])
                            tracker[k].size = np.array(max_bbox[k][2:])

                        else:
                            # 5개의 앵커들의 평균 좌표로 트래킹 설정
                            # color = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [100, 100, 0], [0, 100, 100]]
                            ave_bbox = [0, 0, 0, 0]  # average
                            ave_center = [0, 0]
                            for a in range(len(max_output[k]['anchor'])):
                                cur = max_output[k]['anchor'][a]
                                cur_bbox = cur['bbox']
                                # cur_best_score = cur['best_score']
                                cur_center = [cur['cx'], cur['cy']]
                                cur_bbox = list(map(int, cur_bbox))
                                ave_bbox = [a + b for a, b in zip(ave_bbox, cur_bbox)]
                                ave_center = [a + b for a, b in zip(ave_center, cur_center)]
                            max_bbox[k] = [a // 3 for a in ave_bbox]
                            max_center_bbox[k] = [a / 3 for a in ave_center]
                            tracker[k].center_pos = np.array(max_center_bbox[k])

                        # [속도 측정 단계]
                        if prior_frame_center[k] == (-1, -1):
                            # prior frame center setting: 전체 포컬 플레인 영역을 트래킹한 후 수행
                            prior_frame_center[k] = max_center_bbox[k]
                            prior_distance[k] = get_distance(prior_frame_center[k], first_frame_center[k])

                        else:
                            # current frame center setting: 객체의 전 프레임과 현재 프레임의 이동 거리 차이 계산
                            current_frame_center[k] = max_center_bbox[k]
                            current_distance[k] = get_distance(current_frame_center[k], prior_frame_center[k])

                            if current_distance[k] < prior_distance[k]:  # 속도 감소
                                if max_val[k] > 0.5:  # 정확도가 50프로 이상일때만
                                    necessary_tracking[k] = False  # 다음 프레임을 스킵한다

                            # 다음 프레임 넘어가기전 prior 값 수정
                            prior_frame_center[k] = current_frame_center[k]
                            prior_distance[k] = current_distance[k]

                        end = time.time()
                        print(f"[{k + 1}] update tracking time : {end - start:.2f} sec")

                    else:
                        # 트래커가 현재 프레임을 스킵한다면
                        necessary_tracking[k] = True
                        pass_frame[k] += 1
                        print(f'[{k + 1}] pass tracking frame')

            # Best score 값에 따라 선명도 계산후 영역의 범위를 다르게 설정
            for k in range(tracker_num):
                draw_bboxes(frame, k + 1, max_bbox[k], bbox_color[k])
                if pass_frame[k] > 2:
                    current_target[k] = niqe.sharpness_index(
                        frame_num + 1,
                        focal_range=(start_focal_num, last_focal_num),
                        center=tracker[k].center_pos,
                        count=4)
                    pass_frame[k] = 0

                elif max_val[k] >= 0.8:
                    d_index[k] = 3
                    low_score[k] = 0

                elif 0.2 <= max_val[k] < 0.8:
                    current_target[k] = niqe.sharpness_index(
                        frame_num + 1,
                        focal_range=(start_focal_num, last_focal_num),
                        center=tracker[k].center_pos,
                        count=4)
                    # d_index[k] = 5
                    low_score[k] = 0

                elif 0 <= max_val[k] < 0.2:
                    # d_index[k] = 7
                    low_score[k] += 1
                    if low_score[k] > 1:  # 트래커의 크기가 작아지는 것을 방지
                        tracker[k].size = (max_bbox[k][2] + 1, max_bbox[k][3] + 1)
                    else:
                        current_target[k] = niqe.sharpness_index(
                            frame_num + 1,
                            focal_range=(start_focal_num, last_focal_num),
                            center=tracker[k].center_pos, count=5)

            cv2.imshow(video_name, frame)

            if write_gt:  # gt csv 파일 쓰기
                a_list = []
                for k in range(tracker_num):
                    a_list.extend([max_bbox[k][0], max_bbox[k][1], max_bbox[k][3], max_bbox[k][2]])
                wr.writerow(a_list)

            if is_gt_on:
                gt_bboxes = gt.loc[gt_index].values
                for k in range(tracker_num):
                    gt_bbox = gt_bboxes[4*k:4*(k+1)]
                    iou = IoU(gt_bbox, [max_bbox[k][0], max_bbox[k][1], max_bbox[k][3], max_bbox[k][2]])
                    print(iou, end=' ')
                print()
                gt_index += 1


        if is_record:
            save_image(save_path, frame_num, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

        if frame_num:
            print(f'[{frame_num}] total tracking time : {time.time() - frame_start:.2f} sec')
            print()
            print()

    print(f'total tracking time : {time.time() - total_start:.2f} sec')
    if write_gt:
        gt.close()

if __name__ == "__main__":
    main()
