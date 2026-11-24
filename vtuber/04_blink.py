# [04] 눈 깜빡임 감지  (01 복사 + ★ 부분)
# 실행: python 04_blink.py      종료: 카메라 창 클릭 후 q
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import math                                   # ★

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)                       # 거울처럼 좌우 반전
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # AI 모델은 RGB 순서를 써요
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(image, int(time.time() * 1000))

    if result.face_landmarks:                        # 얼굴을 찾았으면
        face = result.face_landmarks[0]
        pts = [(int(p.x * w), int(p.y * h)) for p in face]   # 0~1 비율 -> 픽셀 좌표
        for p in pts:
            cv2.circle(frame, p, 1, (0, 255, 0), -1)
        # ★★★ 여기부터 새로 치기 ★★★
        eye_l = math.dist(pts[159], pts[145]) / math.dist(pts[33], pts[133])  # 화면 왼쪽 눈
        eye_r = math.dist(pts[386], pts[374]) / math.dist(pts[362], pts[263]) # 화면 오른쪽 눈
        cv2.putText(frame, f"eye {eye_l:.2f} {eye_r:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if eye_l < 0.2 and eye_r < 0.2:
            cv2.putText(frame, "BLINK!", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        # ★★★ 여기까지 ★★★

    cv2.imshow("face", frame)
    if cv2.waitKey(1) == ord("q"):
        break
