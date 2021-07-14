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

    # ground truth
    if is_gt_on:    # IoU 정확도를 측정할 것인지
        f = open('ground_truth/Non_video4_GT.txt', 'r')  # GT 파일

    # create model
    model = ModelBuilder()

    # load model
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(ckpt)
    model.eval().to(device)

    print("Please type the number of trackers: ")
    tracker_num = int(sys.stdin.readline())
    print()

    # build tracker
    tracker = []
    for _ in range(tracker_num):
        tracker.append(build_tracker(model))

    # tracker = build_tracker(model)

    # tracking
    is_first_frame = True
    frame_num = 0
    first_time = True

    current_target = [-1 for _ in range(tracker_num)]
    bbox = [-1 for _ in range(tracker_num)]
    color = [-1 for _ in range(tracker_num)]
    first_frame_center = [(-1, -1) for _ in range(tracker_num)]
    prior_frame_center = [(-1, -1) for _ in range(tracker_num)]
    current_frame_center = [(-1, -1) for _ in range(tracker_num)]
    prior_distance = [-1 for _ in range(tracker_num)]
    current_distance = [-1 for _ in range(tracker_num)]
    necessary_tracking = [True for _ in range(tracker_num)]
    outputs = [-1 for _ in range(tracker_num)]



    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
    for frame, focals in get_frames(video_name, video_type, img2d_ref, start_focal_num, last_focal_num):
        frame_num += 1
        if is_first_frame:
            try:
                for k in range(tracker_num):
                    init_rect = cv2.selectROI(video_name, frame, True, False)
                    build_time = time.time()
                    tracker[k].init(frame, init_rect)
                    print(f"to build tracker time : {time.time() - build_time:.2f} sec")
                    color[k] = list(np.random.random(size=3) * 256)
                print()
            except:
                exit()
            # tracker.init(frame, init_rect)
            is_first_frame = False
        else:
            max_index = [-1 for _ in range(tracker_num)]
            max_val = [-1 for _ in range(tracker_num)]
            max_index_center = [(-1, -1) for _ in range(tracker_num)]

            if first_time:
                print("------ first time ------")
                for k in range(tracker_num):
                    start = time.time()

                    outputs[k] = [tracker[k].track(cv2.imread(f)) for f in focals]

                    for i in range(len(outputs[k])):
                        if outputs[k][i]['best_score'] >= max_val[k]:
                            max_val[k] = outputs[k][i]['best_score']
                            max_index[k] = i

                    end = time.time()
                    print(f"[{k+1}] first tracking time : {end - start:.2f} sec")
                    bbox[k] = list(map(int, outputs[k][max_index[k]]['bbox']))
                    first_frame_center[k] = (outputs[k][max_index[k]]['cx'], outputs[k][max_index[k]]['cy'])

                    current_target[k] = max_index[k]

                first_time = False
                print('------------------------')
            else:
                for k in range(tracker_num):
                    if necessary_tracking[k]:
                        start = time.time()

                        outputs[k] = [tracker[k].track(cv2.imread(focals[i])) for i in range(
                            current_target[k] - 3, current_target[k] + 3)]


                        for i in range(len(outputs[k])):
                            if outputs[k][i]['best_score'] >= max_val[k]:
                                max_val[k] = outputs[k][i]['best_score']
                                max_index[k] = i

                        max_index_center[k] = (outputs[k][max_index[k]]['cx'], outputs[k][max_index[k]]['cy'])


                        if prior_frame_center[k] == (-1, -1):
                            prior_frame_center[k] = max_index_center[k]
                            prior_distance[k] = get_distance(prior_frame_center[k], first_frame_center[k])

                        else:
                            current_frame_center[k] = max_index_center[k]
                            current_distance[k] = get_distance(current_frame_center[k], prior_frame_center[k])

                            if current_distance[k] < prior_distance[k]:  # 속도 감소
                                necessary_tracking[k] = False
                            else:  # 속도 증가
                                prior_frame_center[k] = current_frame_center[k]
                                prior_distance[k] = current_distance[k]


                        end = time.time()
                        print(f"[{k+1}] tracking update time : {end - start:.2f} sec")

                        bbox[k] = list(map(int, outputs[k][max_index[k]]['bbox']))

                        current_target[k] = max_index[k]

                        # if max_index[k] > 3:
                        #     current_target[k] = current_target[k] + abs(3 - max_index[k])
                        # elif max_index[k] < 3:
                        #     current_target[k] = current_target[k] - abs(3 - max_index[k])
                    else:
                        necessary_tracking[k] = True
                        print(f'******* [{k+1}] traker pass frame *******')


            # ground_truth(outputs[max_index]['bbox'][:2],
            #              outputs[max_index]['bbox'][2:])
            for k in range(tracker_num):
                draw_bboxes(frame, k+1, bbox[k], color[k])
            print()

            # save_path = os.path.join(
            #     'data/result2', '{:03d}.jpg'.format(frame_num))
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

            cv2.imshow(video_name, frame)

            if is_record:
                save_image(save_path, frame_num, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

if __name__ == "__main__":
    main()
