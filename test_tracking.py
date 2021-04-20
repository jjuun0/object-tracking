import cv2
# import sys

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
# print(cv2.__version__)  # opencv-contrib-python == 3.4.2.16


def define_tracker():
    """ tracker 를 설정하는 함수 """

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[-1]
    tracker = 0

    if int(minor_ver) < 3:  # opencv 가 4.0 버전 이상일 경우
        tracker = cv2.Tracker_create(tracker_type)
    else:
        # 8개의 다른 tracker 를 사용할 수 있다.
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    return tracker_type, tracker


def whole_focal_tracking(tracker_type, tracker, image_folder):
    """ 모든 프레임의 focal 사진 파일에서 tracking 하는 함수 """

    # for new_frame in range(91):
    for new_frame in range(3):

        frame_numbering = str(new_frame).zfill(3)  # 000, 001, 002, ,,,
        new_frame_folder = image_folder + frame_numbering + '/focal/'
        # file = ["focal", "images"]  # 2D 객체 추적 : images

        for focal_num in range(100, 0, -1):
            focal_numbering = str(focal_num).zfill(3)
            new_image_address = new_frame_folder + focal_numbering + '.png'

            # Read a new frame
            new_image = cv2.imread(new_image_address)

            # Update tracker
            ok, bbox = tracker.update(new_image)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(new_image, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(new_image, "Tracking failure detected", (130, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)

            cv2.putText(new_image, str(new_frame) + " frame", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 50, 200), 2)

            # Display tracker type on frame
            cv2.putText(new_image, tracker_type + " Tracker", (130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 170, 50), 2)

            # Display result
            cv2.imshow("Tracking", new_image)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break


def camera_image_tracking(tracker_type, tracker, image_folder, camera):
    """ 모든 프레임의 images 폴더 에서 몇 번째 camera 로 2D tracking 하는 함수 """

    for new_frame in range(1, 91):

        frame_numbering = str(new_frame).zfill(3)  # 000, 001, 002, ,,,
        new_frame_folder = image_folder + frame_numbering + '/images/'

        camera_numbering = str(camera).zfill(3)
        new_image_address = new_frame_folder + camera_numbering + '.png'

        # Read a new frame
        new_image = cv2.imread(new_image_address)
        # cropped_img = new_image[y: y + h, x: x + w]

        # Update tracker
        ok, bbox = tracker.update(new_image)
        # ok, bbox = tracker.update(cropped_img)

        # Draw bounding box
        if ok:
            # Tracking success
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # x, y : 좌상단 위치 / w, h : 가로, 세로 크기

        else:
            # Tracking failure
            cv2.putText(new_image, "Tracking failure detected", (130, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(new_image, str(new_frame) + " frame", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 200), 2)

        # Display tracker type on frame
        cv2.putText(new_image, tracker_type + " Tracker", (130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", new_image)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


if __name__ == '__main__':

    my_tracker_type, my_tracker = define_tracker()

    image = cv2.imread('NonVideo3_tiny/000/images/005.png')

    # Define a bounding box
    # goturn 사용시 자동으로 객체 검출하기 위해 초기화 시켜줌.
    # bbox = (276, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    # Drag rectangle from top left to bottom right
    bbox = cv2.selectROI(image, False)

    # Initialize tracker with first frame and bounding box
    # my_tracker.init(image, init_bbox)
    my_tracker.init(image, bbox)

    my_image_folder = 'NonVideo3_tiny/'

    # whole_focal_tracking(my_tracker_type, my_tracker, my_image_folder)

    camera_image_tracking(my_tracker_type, my_tracker, my_image_folder, 5)
