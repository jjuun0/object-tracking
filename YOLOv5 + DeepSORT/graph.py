import numpy as np
import matplotlib.pyplot as plt

file = open('D:/dataset/NonVideo3_tiny_result/YOLOv5 + DeepSORT/result/result.txt', 'r')
frame_list = []  # 20 : 24.723
frame_list_index = -1
while True:
    line = file.readline()
    if line.startswith('Done'):  # Done. (1472.480s)
        break

    if line.startswith('['):
        line = line.split()
        if line[1] == 'total':
            frame_list_index += 1
            frame_dict = {}
            frame = int(line[0][1:-1])
            sec = float(line[-2])
            frame_dict[frame] = sec
            frame_list.append(frame_dict)

    # elif line[:-1].endswith('s)'):  # 544x960 Done. (0.459s)
    #     sec = float(line.split()[-1][1:-2])
    #     frame_list[frame_list_index][frame][0] += sec
    #     frame_list[frame_list_index][frame][1] += 1
        # print(sec)

    # print(line)
print(frame_list)


# graph
x = []
y = []

for frame in frame_list:
    frame, sec = list(frame.items())[0]
    x.append(frame)
    y.append(sec)

y_average = sum(y) / len(y)
y_average = [y_average] * len(y)

plt.plot(x, y, marker="o", label="Frame")
plt.plot(x, y_average, label="Average")

plt.xlabel("Frame")
plt.ylabel("SEC")
plt.title("Frame average in Focal planes by YOLOv5 + DeepSORT")
plt.legend()
plt.show()





