# [03] 입 벌림을 숫자로  (01 복사 + ★ 부분)
# 실행: python 03_mouth_open.py      종료: 카메라 창 클릭 후 q
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
        mouth = math.dist(pts[13], pts[14]) / math.dist(pts[61], pts[291])   # 입 높이 / 입 너비
        print(f"입 벌림 {mouth:.2f}", "#" * int(mouth * 40))
        cv2.putText(frame, f"mouth {mouth:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # ★★★ 여기까지 ★★★

    cv2.imshow("face", frame)
    if cv2.waitKey(1) == ord("q"):
        break
