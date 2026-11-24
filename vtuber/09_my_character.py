# [09] 내 캐릭터로 메쉬 만들기 + '점' 을 돌려서 고개 기울이기  (08 복사 + ★ 부분)
#   준비: python make_rig.py my_char/character.png  로 my_char/rig.json 만들기
#   07 은 그림 전체를 돌렸어요. 이번엔 얼굴 '점' 만 돌리고, 그림은 삼각형이 따라 늘어나요 -> 몸은 그대로!
# 실행: python 09_my_character.py      종료: 카메라 창 클릭 후 q
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import math
import numpy as np
import json                                          # ★

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# ★★★ 08 의 char ~ LOWER, CHIN 줄을 지우고 여기부터 새로 치기 ★★★
rig = json.load(open("my_char/rig.json"))            # make_rig.py 로 찍은 점들
char = cv2.imread("my_char/" + rig["image"], cv2.IMREAD_UNCHANGED)
if char.shape[2] == 3:                               # 투명 배경이 없는 그림이면 투명도 채널 추가
    char = cv2.cvtColor(char, cv2.COLOR_BGR2BGRA)
PAD = int(char.shape[1] * 0.15)                      # 둘레에 투명 여백 (머리가 돌아갈 공간)
char = cv2.copyMakeBorder(char, PAD, 0, PAD, PAD, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
S = 700 / char.shape[0]                              # 창 높이 700 에 맞게 줄이기
char = cv2.resize(char, None, fx=S, fy=S)
H, W = char.shape[:2]
NAMES = list(rig["points"])                          # 점 이름 27개 (eye_l_a, nose, chin, ...)
I = {n: i for i, n in enumerate(NAMES)}              # 이름 -> 번호
HAIR = [e["pos"] for e in rig["extra"] if e["follow"] > 0]     # make_rig 좌클릭 점 (머리카락: 머리를 따라감)
BODY = [e["pos"] for e in rig["extra"] if e["follow"] == 0]    # make_rig 우클릭 점 (몸: 고정)
PTS = (np.float32([rig["points"][n] for n in NAMES] + HAIR) + PAD) * S      # 여백만큼 밀고, 줄인 만큼 줄이기
FACE = len(PTS)                                      # 이 번호 전까지가 머리와 함께 움직이는 점
PTS = np.vstack([PTS, (np.float32(BODY).reshape(-1, 2) + PAD) * S,
                 [[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]]])   # 몸 점 + 모서리 4개 (고정)
PTS[I["mouth_lower"], 1] += 4                        # 윗입술과 같은 자리면 삼각형이 안 생겨서 살짝 띄우기
MOUTH_W = math.dist(PTS[I["mouth_l"]], PTS[I["mouth_r"]])
base = None
# ★★★ 여기까지 ★★★

sub = cv2.Subdiv2D((0, 0, W, H))                     # 점들을 삼각형으로 잇기 (들로네 삼각분할)
for x, y in PTS:
    sub.insert((float(x), float(y)))
TRIS = []
for t in sub.getTriangleList().reshape(-1, 3, 2):
    ids = [int(np.argmin(((PTS - v) ** 2).sum(1))) for v in t]
    if ((PTS[ids] - t) ** 2).sum() < 1:
        TRIS.append(ids)


def warp_triangle(src, dst, out):
    """원래 삼각형 src 를 새 삼각형 dst 로 늘여 붙이기 (3점 -> 3점 변환)"""
    x, y, w, h = cv2.boundingRect(np.int32(dst))
    M = cv2.getAffineTransform(np.float32(src), np.float32(dst - [x, y]))
    piece = cv2.warpAffine(char, M, (w, h))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst - [x, y]), 255)
    cv2.copyTo(piece, mask, out[y:y + h, x:x + w])


while True:
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(image, int(time.time() * 1000))

    D = PTS.copy()
    if result.face_landmarks:
        face = result.face_landmarks[0]
        pts = [(int(p.x * w), int(p.y * h)) for p in face]
        for p in pts:
            cv2.circle(frame, p, 1, (0, 255, 0), -1)
        # ★★★ 08 의 입 부분을 지우고 여기부터 새로 치기 ★★★
        ratio = math.dist(pts[13], pts[14]) / math.dist(pts[61], pts[291])
        drop = min(max((ratio - 0.05) / 0.5, 0), 1) * MOUTH_W * 0.5          # 입 벌린 정도 -> 내려갈 픽셀
        for name, k in (("mouth_lower", 1), ("mouth_bottom", 1), ("chin", 0.55), ("jaw_l", 0.25), ("jaw_r", 0.25)):
            D[I[name], 1] += drop * k                                        # 아랫입술 > 턱 > 턱선 순서로 덜 내려감

        tilt = math.degrees(math.atan2(pts[263][1] - pts[33][1], pts[263][0] - pts[33][0]))
        if base is None:
            base = tilt
        a = math.radians(tilt - base)
        neck = PTS[I["neck"]]
        rot = np.float32([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        D[:FACE] = neck + (D[:FACE] - neck) @ rot                            # 얼굴 점만 목 기준으로 회전
        D = np.clip(D, 0, [W - 1, H - 1])                                    # 그림 밖으로 안 나가게
        # ★★★ 여기까지 ★★★

    out = char.copy()
    for t in TRIS:
        if (D[t] != PTS[t]).any():
            warp_triangle(PTS[t], D[t], out)
    a = out[:, :, 3:] / 255
    canvas = (out[:, :, :3] * a + np.array([0, 255, 0]) * (1 - a)).astype(np.uint8)
    if np.linalg.norm(D[I["mouth_lower"]] - D[I["mouth_upper"]]) > 8:         # ★ 입 안 칠하기 (점 이름으로)
        cv2.fillPoly(canvas, [np.int32(D[[I["mouth_l"], I["mouth_upper"], I["mouth_r"], I["mouth_lower"]]])], (50, 35, 100))
    for t in TRIS:
        cv2.polylines(canvas, [np.int32(D[t])], True, (255, 200, 0), 1)
    cv2.imshow("vtuber", canvas)

    cv2.imshow("face", frame)
    if cv2.waitKey(1) == ord("q"):
        break
