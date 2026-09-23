#!/usr/bin/env python3
import cv2, time
cap = cv2.VideoCapture('/dev/video21', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'摄像头: {rw}x{rh}')
cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Preview', 960, 540)
t0 = time.perf_counter(); fc = 0
while True:
    ret, frame = cap.read()
    if not ret: time.sleep(0.005); continue
    fc += 1
    if fc % 30 == 0:
        el = time.perf_counter() - t0
        print(f'{fc} 帧, {fc/el:.1f} fps')
    disp = cv2.resize(frame, (960, 540))
    cv2.imshow('Preview', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
