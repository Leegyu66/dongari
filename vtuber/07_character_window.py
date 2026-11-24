# [07] 내 얼굴 = 리모컨 - 기울기·크기 숫자로 캐릭터 그림 전체 돌리기  (01 복사 + ★ 부분)
#   캐릭터는 내 얼굴에 붙이지 않고 따로 창에 띄워요. 내 얼굴에서는 '숫자' 만 가져옵니다.
# 실행: python 07_character_window.py      종료: 카메라 창 클릭 후 q
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import math                                          # ★
import numpy as np                                   # ★

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# ★★★ 여기부터 새로 치기 ★★★
char = cv2.imread("character.png", cv2.IMREAD_UNCHANGED)    # 캐릭터 그림 (투명 배경)
H, W = char.shape[:2]
NECK = (350, 665)                                   # 캐릭터 그림 속 목 위치 = 회전 중심
base = None                                         # 기준 얼굴 (처음 잡힌 내 얼굴)
M = cv2.getRotationMatrix2D(NECK, 0, 1)             # 처음엔 그대로
# ★★★ 여기까지 ★★★

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
        tilt = math.degrees(math.atan2(pts[263][1] - pts[33][1], pts[263][0] - pts[33][0]))  # 05번 기울기
        size = math.dist(pts[33], pts[263])                    # 두 눈 사이 거리 (가까이 가면 커짐)
        if base is None:
            base = (tilt, size)                                # 처음 얼굴을 기준으로 저장
        M = cv2.getRotationMatrix2D(NECK, -(tilt - base[0]), size / base[1])   # 목 기준으로 회전 + 확대
        # ★★★ 여기까지 ★★★

    # ★★★ 여기부터 새로 치기: 캐릭터 창 그리기 ★★★
    warped = cv2.warpAffine(char, M, (W, H))                   # 그림 전체를 M 대로 움직이기
    a = warped[:, :, 3:] / 255                                 # 투명도
    canvas = (warped[:, :, :3] * a + np.array([0, 255, 0]) * (1 - a)).astype(np.uint8)  # 초록 배경 위에
    cv2.imshow("vtuber", canvas)
    # ★★★ 여기까지 ★★★

    cv2.imshow("face", frame)
    if cv2.waitKey(1) == ord("q"):
        break
