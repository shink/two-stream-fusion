# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: demo.py
@date: 2022/3/10
@description: 
"""
import os.path

import numpy as np
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture("C:\\Users\\shenke\\Desktop\\Fighting030_x264.mp4")

    # # 角点检测参数
    # feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
    #
    # # KLT光流参数
    # lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))
    #
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # # out = cv2.VideoWriter("reslut.avi", cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
    # # (np.int(width), np.int(height)), True)
    #
    # tracks = []
    # track_len = 15
    # frame_idx = 0
    # detect_interval = 5
    #
    # count = 0
    #
    # while True:
    #
    #     ret, frame = cap.read()
    #     if ret:
    #         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         vis = frame.copy()
    #
    #         if len(tracks) > 0:
    #             img0, img1 = prev_gray, frame_gray
    #             p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    #             # 上一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
    #             p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    #
    #             # 反向检查,当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
    #             p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    #
    #             # 得到角点回溯与前一帧实际角点的位置变化关系
    #             d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    #
    #             # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
    #             good = d < 1
    #
    #             new_tracks = []
    #
    #             for i, (tr, (x, y), flag) in enumerate(zip(tracks, p1.reshape(-1, 2), good)):
    #
    #                 # 判断是否为正确的跟踪点
    #                 if not flag:
    #                     continue
    #
    #                 # 存储动态的角点
    #                 tr.append((x, y))
    #
    #                 # 只保留track_len长度的数据，消除掉前面的超出的轨迹
    #                 if len(tr) > track_len:
    #                     del tr[0]
    #                 # 保存在新的list中
    #                 new_tracks.append(tr)
    #
    #                 cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    #
    #             # 更新特征点
    #             tracks = new_tracks
    #
    #             # #以上一振角点为初始点，当前帧跟踪到的点为终点,画出运动轨迹
    #             cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0), 1)
    #
    #         # 每隔 detect_interval 时间检测一次特征点
    #         if frame_idx % detect_interval == 0:
    #             mask = np.zeros_like(frame_gray)
    #             mask[:] = 255
    #
    #             if frame_idx != 0:
    #                 for x, y in [np.int32(tr[-1]) for tr in tracks]:
    #                     cv2.circle(mask, (x, y), 5, 0, -1)
    #
    #             p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
    #             if p is not None:
    #                 for x, y in np.float32(p).reshape(-1, 2):
    #                     tracks.append([(x, y)])
    #
    #         frame_idx += 1
    #         prev_gray = frame_gray
    #
    #         vis = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
    #         cv2.imshow('track', vis)
    #         # # out.write(vis)
    #         # ch = cv2.waitKey(1)
    #         # if ch == 27:
    #         #     cv2.imwrite('track.jpg', vis)
    #         #     break
    #
    #         cv2.imwrite(os.path.join("C:\\Users\\shenke\\Desktop\\2", '%.6d.jpg' % count), vis)
    #
    #         count += 1
    #
    #     else:
    #         break
    #
    # cv2.destroyAllWindows()
    # cap.release()

    # 获取第一帧
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    # 遍历每一行的第1列
    hsv[..., 1] = 255

    count = 0

    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 返回一个两通道的光流向量，实际上是每个点的像素位移值
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # print(flow.shape)
        print(flow)

        # 笛卡尔坐标转换为极坐标，获得极轴和极角
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png', frame2)
        #     cv2.imwrite('opticalhsv.png', rgb)

        cv2.imwrite(os.path.join("C:\\Users\\shenke\\Desktop\\4", '%.6d.jpg' % count), rgb)
        count += 1
        prvs = next

    cap.release()
    cv2.destroyAllWindows()
