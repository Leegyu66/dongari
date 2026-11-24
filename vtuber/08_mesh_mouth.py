# [08] 메쉬 맛보기 - 그림을 삼각형으로 쪼개고, 입 벌리면 아랫입술·턱 점만 내리기  (01 복사 + ★ 부분)
#   07 은 그림 '전체' 를 한 번에 움직였어요. 입만 벌리려면? -> 그림을 삼각형 조각으로 나누고 조각마다 따로!
# 실행: python 08_mesh_mouth.py      종료: 카메라 창 클릭 후 q
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
char = cv2.imread("character.png", cv2.IMREAD_UNCHANGED)
H, W = char.shape[:2]
PTS = np.float32([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1],      # 0~3: 그림 네 모서리 (고정)
                  [233, 453], [425, 409], [297, 522], [350, 665],        # 4~7: 왼눈, 오른눈, 코, 목
                  [203, 512], [487, 506], [231, 587], [450, 569],        # 8~11: 왼볼, 오른볼, 왼턱선, 오른턱선
                  [316, 566], [378, 553], [340, 575], [340, 581],        # 12~15: 입꼬리 L, R, 윗입술, 아랫입술
                  [350, 628]])                                           # 16: 턱
LOWER, CHIN = 15, 16

sub = cv2.Subdiv2D((0, 0, W, H))                     # 점들을 삼각형으로 잇기 (들로네 삼각분할)
for x, y in PTS:
    sub.insert((float(x), float(y)))
TRIS = []
for t in sub.getTriangleList().reshape(-1, 3, 2):
    ids = [int(np.argmin(((PTS - v) ** 2).sum(1))) for v in t]   # 삼각형 꼭짓점 -> 점 번호
    if ((PTS[ids] - t) ** 2).sum() < 1:                          # 그림 밖 가짜 삼각형은 빼기
        TRIS.append(ids)


def warp_triangle(src, dst, out):
    """원래 삼각형 src 를 새 삼각형 dst 로 늘여 붙이기 (3점 -> 3점 변환)"""
    x, y, w, h = cv2.boundingRect(np.int32(dst))                 # 새 삼각형을 감싸는 사각형
    M = cv2.getAffineTransform(np.float32(src), np.float32(dst - [x, y]))
    piece = cv2.warpAffine(char, M, (w, h))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst - [x, y]), 255)        # 삼각형 모양만
    cv2.copyTo(piece, mask, out[y:y + h, x:x + w])
# ★★★ 여기까지 ★★★

while True:
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)                       # 거울처럼 좌우 반전
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # AI 모델은 RGB 순서를 써요
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(image, int(time.time() * 1000))

    D = PTS.copy()                                   # ★ 이번 프레임의 점 위치 (처음엔 그대로)
    if result.face_landmarks:                        # 얼굴을 찾았으면
        face = result.face_landmarks[0]
        pts = [(int(p.x * w), int(p.y * h)) for p in face]   # 0~1 비율 -> 픽셀 좌표
        for p in pts:
            cv2.circle(frame, p, 1, (0, 255, 0), -1)
        # ★★★ 여기부터 새로 치기 ★★★
        ratio = math.dist(pts[13], pts[14]) / math.dist(pts[61], pts[291])   # 03번 입 벌림
        drop = min(ratio, 0.6) * 60                  # 아랫입술이 내려갈 픽셀
        D[LOWER, 1] += drop                          # 아랫입술 내리기
        D[CHIN, 1] += drop * 0.6                     # 턱은 조금 덜
        # ★★★ 여기까지 ★★★

    # ★★★ 여기부터 새로 치기: 움직인 삼각형만 다시 붙이기 ★★★
    out = char.copy()
    for t in TRIS:
        if (D[t] != PTS[t]).any():
            warp_triangle(PTS[t], D[t], out)
    a = out[:, :, 3:] / 255
    canvas = (out[:, :, :3] * a + np.array([0, 255, 0]) * (1 - a)).astype(np.uint8)
    if D[LOWER, 1] - D[14, 1] > 8:                   # 입이 벌어졌으면 입 안 칠하기
        cv2.fillPoly(canvas, [np.int32(D[[12, 14, 13, 15]])], (50, 35, 100))
    for t in TRIS:                                   # 메쉬 보기 (지우면 안 보여요)
        cv2.polylines(canvas, [np.int32(D[t])], True, (255, 200, 0), 1)
    cv2.imshow("vtuber", canvas)
    # ★★★ 여기까지 ★★★

    cv2.imshow("face", frame)
    if cv2.waitKey(1) == ord("q"):
        break
