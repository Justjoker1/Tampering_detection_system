from tabnanny import verbose
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict


#####################
# Utility Functions #
#####################

def do_intersect(p1, p2, p3, p4):
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if np.isclose(val, 0):
            return 0
        return 1 if val > 0 else 2

    o1, o2 = orientation(p1, p2, p3), orientation(p1, p2, p4)
    o3, o4 = orientation(p3, p4, p1), orientation(p3, p4, p2)
    return o1 != o2 and o3 != o4


def get_collided_sides(x, y, w, h, polygon):
    left, right = x - w / 2, x + w / 2
    top, bottom = y - h / 2, y + h / 2
    bbox_edges = [
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top))
    ]
    collided = set()
    n = len(polygon)
    for i in range(n):
        p1 = tuple(polygon[i])
        p2 = tuple(polygon[(i + 1) % n])
        for edge in bbox_edges:
            if do_intersect(edge[0], edge[1], p1, p2):
                collided.add(i)
    return collided


def is_point_in_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), measureDist=False) >= 0


def compute_speed(track, fps, pixel_length_df):
    if len(track) > 1:
        x1, y1 = track[-2]
        x2, y2 = track[-1]
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        x2_idx = int(np.clip(x2, 0, pixel_length_df.shape[1] - 1))
        y2_idx = int(np.clip(y2, 0, pixel_length_df.shape[0] - 1))
        length_per_pixel = pixel_length_df.at[y2_idx, x2_idx]
        real_distance = pixel_distance * length_per_pixel
        speed_kph = real_distance * (fps / 2) * 3600
        return round(speed_kph, 1)
    return 0


#############################
# 动态绘图封装（调试使用）  #
#############################

def setup_dynamic_plot():
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    line1, = axs[0].plot([], [], marker='o', linestyle='-', color='b')
    axs[0].set_ylabel('新增突然消失车辆数')
    axs[0].set_title('实时新增突然消失车辆统计')
    axs[0].grid()
    line2, = axs[1].plot([], [], marker='o', linestyle='-', color='r')
    axs[1].set_xlabel('帧号')
    axs[1].set_ylabel('新增突然出现车辆数')
    axs[1].set_title('实时新增突然出现车辆统计')
    axs[1].grid()
    return fig, axs, line1, line2


def update_dynamic_plot(fig, axs, line1, line2, frame_indices, disappeared_counts, appeared_counts):
    line1.set_xdata(frame_indices)
    line1.set_ydata(disappeared_counts)
    line2.set_xdata(frame_indices)
    line2.set_ydata(appeared_counts)
    for ax in axs:
        ax.relim()
        ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


#####################################
# 特征提取模块：extract_features()  #
#####################################

def extract_features(video_path,
                     result_path="../output/video/result_demo.mp4",
                     show_plot=False,
                     show_text=False,
                     save_video=True,
                     report=False,
                     show_video=False,):
    model = YOLO('./features_extractor/yolov10m.pt')
    polygonPoints = np.array([[40, 360], [1230, 360], [1230, 600], [40, 600]], dtype=np.int32)

    track_history = defaultdict(list)
    speed_cache = {}
    vehicle_status = {}   # tid -> (collision_count, status)
    baseframe = pd.DataFrame(columns=["ID", "Speed", "Trajectory", "Status", "Frame_Range"])
    newly_disappeared_counts = []
    newly_appeared_counts = []
    frame_indices = []
    event_counts_df = pd.DataFrame(columns=["Frame", "Disappeared", "Appeared"])

    # 构造像素长度映射
    image_height, image_width = 720, 1280
    known_y = np.array([360, 600])
    known_length_per_pixel = np.array([(3 * 3.7) / (180 * 1000), (3 * 3.7) / (500 * 1000)])
    y_indices = np.arange(image_height)
    conversion_factors = np.interp(y_indices, known_y, known_length_per_pixel)
    pixel_length_array = np.tile(conversion_factors.reshape(-1, 1), (1, image_width))
    pixel_length_df = pd.DataFrame(pixel_length_array)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    RESIZE_WIDTH, RESIZE_HEIGHT = 1280, 720
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWriter = None
    frame_count = 0

    if show_plot:
        fig, axs, line1, line2 = setup_dynamic_plot()
    else:
        fig = axs = line1 = line2 = None

    def initialize_target(tid, track, frame_count, init_status,in_poly):
        vehicle_status[tid] = (1 if init_status == 0 else 0, init_status)
        if not in_poly:
            frame_count = None
        new_row = {"ID": tid, "Speed": [], "Trajectory": track,
                   "Status": init_status, "Frame_Range": [frame_count, None]}
        nonlocal baseframe
        baseframe = pd.concat([baseframe, pd.DataFrame([new_row])], ignore_index=True)

    def update_detection(box, tid, frame_count):
        x, y, w, h = box
        in_poly = is_point_in_polygon(int(x), int(y), polygonPoints)
        if in_poly:
            """"""
            track_history[tid].append((int(x), int(y)))
        curr_track = track_history[tid]
        if frame_count % 2 == 0:
            spd = compute_speed(curr_track, fps, pixel_length_df)
            speed_cache[tid] = spd
        else:
            spd = speed_cache.get(tid, 0)
        collided = get_collided_sides(x, y, w, h, polygonPoints)

        return curr_track, spd, collided, in_poly

    def update_target_state(tid, collided, in_poly, frame_count):
        collision_count, status = vehicle_status[tid]
        # —— 处理首次碰撞，多边形内进入——
        if frame_count > 1 and status is None and collision_count == 0:
            if collided:
                vehicle_status[tid] = (collision_count + 1, 0)
                status = 0
            elif in_poly:
                vehicle_status[tid] = (collision_count, 2)
                status = 2
            """"""
            idx = baseframe[baseframe['ID'] == tid].index[0]
            baseframe.at[idx, 'Frame_Range'][0] = frame_count
        # —— 处理正常驶出：只有 collision_count==1 且 status==0 时才写一次 ——
        if collided and status == 0 and collision_count == 1:
            # 更新到“驶出”状态
            vehicle_status[tid] = (collision_count + 1, 3)
            status = 3
            # 写结束帧（仅第一次）
            idx = baseframe[baseframe['ID'] == tid].index[0]
            start_f = baseframe.at[idx, 'Frame_Range'][0]
            baseframe.at[idx, 'Frame_Range'] = (start_f, frame_count)
        if collided and status == 2:
            # 更新到“驶出”状态
            vehicle_status[tid] = (collision_count + 1, 2)
            status = 2
            # 写结束帧（仅第一次）
            idx = baseframe[baseframe['ID'] == tid].index[0]
            start_f = baseframe.at[idx, 'Frame_Range'][0]
            baseframe.at[idx, 'Frame_Range'] = (start_f, frame_count)
        if collided and status == 3:
            # 更新到“驶出”状态
            status = 3
            # 写结束帧（仅第一次）
            idx = baseframe[baseframe['ID'] == tid].index[0]
            start_f = baseframe.at[idx, 'Frame_Range'][0]
            baseframe.at[idx, 'Frame_Range'] = (start_f, frame_count)
        return status

    def update_baseframe(tid, track, spd, status, in_poly):
        idx = baseframe[baseframe['ID'] == tid].index[0]
        # 更新轨迹和状态保持不变...
        baseframe.at[idx, 'Trajectory'] = track
        baseframe.at[idx, 'Status'] = status
        """"""
        if in_poly:
            if spd:
                lst = baseframe.at[idx, 'Speed']
                if len(lst) == 0:
                    # 第一次非零速度来临时，用同样的值填充“第一帧”
                    lst.append(spd)  # 这是第二帧的速度
                    lst.insert(0, spd)  # 复制到第一帧
                else:
                    lst.append(spd)
                baseframe.at[idx, 'Speed'] = lst
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.polylines(frame, [polygonPoints], True, (0, 0, 255), 3)

        results = model.track(frame, persist=True, classes=[2,5,7], verbose=report)
        a_frame = results[0].plot()
        if results[0].boxes.xywh.numel() == 0:
            boxes, track_ids = [], []
        else:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

        detected_ids = set()
        newly_disappeared = 0
        newly_appeared = 0

        # 处理检测到的
        for box, tid in zip(boxes, track_ids):
            detected_ids.add(tid)
            curr_track, spd, collided, in_poly = update_detection(box, tid, frame_count)
            if tid not in vehicle_status:
                init_status = 0 if in_poly and frame_count <= 1 else None
                initialize_target(tid, curr_track, frame_count, init_status,in_poly)
                if init_status == 2:
                    newly_appeared += 1
            else:
                prev_status = vehicle_status[tid][1]
                new_status = update_target_state(tid, collided, in_poly, frame_count)
                if prev_status is None and new_status == 2:
                    newly_appeared += 1

            update_baseframe(tid, curr_track, spd, vehicle_status[tid][1], in_poly)

            # 可视化轨迹与速度
            pts = np.array(track_history[tid], np.int32).reshape(-1,1,2)
            cv2.polylines(a_frame, [pts], False, (255,0,255), 3)
            cv2.putText(a_frame, f"{spd:.1f} km/h", (int(box[0]), int(box[1]-20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 处理未检测到的：只在 status==0 时写消失帧
        for tid in list(vehicle_status):
            if tid not in detected_ids:
                collision_count, status = vehicle_status[tid]
                if status == 0:
                    # 状态改为突然消失(1)，并写结束帧
                    vehicle_status[tid] = (collision_count, 1)
                    newly_disappeared += 1
                    idx = baseframe[baseframe['ID'] == tid].index[0]
                    start_f = baseframe.at[idx, 'Frame_Range'][0]
                    baseframe.at[idx, 'Frame_Range'] = (start_f, frame_count)

        newly_disappeared_counts.append(newly_disappeared)
        newly_appeared_counts.append(newly_appeared)
        frame_indices.append(frame_count)

        # 记录帧事件
        event_counts_df = pd.concat([event_counts_df,
            pd.DataFrame([{"Frame":frame_count,
                           "Disappeared":newly_disappeared,
                           "Appeared":newly_appeared}])],
            ignore_index=True)

        if save_video:
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                videoWriter = cv2.VideoWriter(
                    result_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (RESIZE_WIDTH, RESIZE_HEIGHT)  # <- 和 resize 后的 frame 大小保持一致
                )
            videoWriter.write(a_frame)
        if show_video:
            cv2.imshow("Track", a_frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

        frame_count += 1
        if show_plot:
            update_dynamic_plot(fig, axs, line1, line2,
                                frame_indices,
                                newly_disappeared_counts,
                                newly_appeared_counts)

    cap.release()
    if videoWriter:
        videoWriter.release()
    cv2.destroyAllWindows()
    if show_plot:
        import matplotlib.pyplot as plt
        plt.ioff()

    # 清洗 DataFrame
    #baseframe = baseframe[baseframe['Speed'].apply(lambda x: len(x)>0)]
    baseframe = baseframe.dropna(subset=['Status'])

    if show_text:
        print("状态说明：\n"
              "0  —— 正常驶入：目标按预期方式进入监控区域（与区域边界发生碰撞），\n"
              "1  —— 突然消失：原处于正常驶入状态的目标后续未被检测到，视为异常消失，\n"
              "2  —— 突然出现：目标在非初始帧中突然出现在区域内，无正常的边界碰撞过程，\n"
              "3  —— 正常驶出：目标在正常驶入后，再次发生边界碰撞，视为依正常方式离开区域，\n"
              "None —— 未归类/未确定：目标初始状态不明确，尚无足够信息判断行为。")
        print("总突然消失次数：", sum(newly_disappeared_counts))
        print("总突然出现次数：", sum(newly_appeared_counts))
        print("Baseframe 前 5 行：")
        print(baseframe)
        baseframe = baseframe.drop(columns=['Trajectory'])
    save_folder1 = "./output/features_temp/features"
    save_folder2 = "./output/features_temp/events"
    baseframe.to_csv(os.path.join(save_folder1, 'baseframe.csv'), index=False)
    # 保存 event_counts_df 到 xlsx 文件
    event_counts_df.to_csv(os.path.join(save_folder2, 'baseframe_event.csv'), index=False)
    return baseframe, event_counts_df


#####################################
# 模块调用示例
#####################################
if __name__ == "__main__":
    video_path = r".\dataset_split\train\video\ori01.mp4"
    baseframe, event_counts_df = extract_features(video_path, show_plot=False, show_text=True)
