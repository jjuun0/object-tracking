def IoU(box1, box2):
    # box : x, y, h, w
    box1_area = box1[2]*box1[3]
    box2_area = box2[2]*box2[3]

    # x2, y2 in box1, box2
    box1_x2, box1_y2 = box1[0]+box1[3], box1[1] + box1[2]
    box2_x2, box2_y2 = box2[0]+box2[3], box2[1] + box2[2]

    # overlapping region
    overlap_x1, overlap_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    overlap_x2, overlap_y2 = min(box1_x2, box2_x2), min(box1_y2, box2_y2)
    overlap_area = (overlap_x2-overlap_x1) * (overlap_y2-overlap_y1)

    # calculate Iou
    return f'{overlap_area / (box1_area + box2_area - overlap_area):.3f}'


if __name__ == '__main__':
    box1 = [10, 20, 20, 20]
    box2 = [20, 10, 20, 20]
    iou = IoU(box1, box2)
    print(iou)
