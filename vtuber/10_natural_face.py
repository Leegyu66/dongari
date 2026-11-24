# [10] 완성판 - 메쉬 변형 버튜버 (카메라 창 + 캐릭터 창)
#   1) 캐릭터 그림을 점(rig.json)으로 삼각형 그물(메쉬)로 쪼개고
#   2) 내 얼굴 숫자에 따라 점을 움직인 뒤
#   3) 삼각형마다 그림을 늘이고 줄여서 붙입니다.  + 눈꺼풀·입 안은 덧그리기
# 실행: python 10_natural_face.py      q 종료 / c 기준 얼굴 다시 잡기 / w 메쉬 보기
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import math
import numpy as np
import json

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True,                    # AI 표정 점수 받기 (06번)
)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# =============== 1. 캐릭터 불러오기 ===============
FOLDER = "my_char/"
rig = json.load(open(FOLDER + "rig.json"))
char = cv2.imread(FOLDER + rig["image"], cv2.IMREAD_UNCHANGED)
if char.shape[2] == 3:
    char = cv2.cvtColor(char, cv2.COLOR_BGR2BGRA)
PAD_X, PAD_Y = int(char.shape[1] * 0.2), int(char.shape[0] * 0.1)   # 둘레에 투명 여백 (머리가 움직일 공간)
char = cv2.copyMakeBorder(char, PAD_Y, 0, PAD_X, PAD_X, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
FIT = 720 / char.shape[0]                            # 창 높이 720 에 맞추기
char = cv2.resize(char, None, fx=FIT, fy=FIT, interpolation=cv2.INTER_AREA)
H, W = char.shape[:2]
P = {k: (np.array(v, float) + [PAD_X, PAD_Y]) * FIT for k, v in rig["points"].items()}
BG = (0, 255, 0)

EYE_L = np.mean([P[f"eye_l_{k}"] for k in ("a", "b", "top", "bottom")], axis=0)
EYE_R = np.mean([P[f"eye_r_{k}"] for k in ("a", "b", "top", "bottom")], axis=0)
RIGHT = (EYE_R - EYE_L) / np.linalg.norm(EYE_R - EYE_L)     # 캐릭터 얼굴의 '오른쪽' 방향
UP = np.array([RIGHT[1], -RIGHT[0]])                        # 캐릭터 얼굴의 '위' 방향
FACE_W = math.dist(P["face_l"], P["face_r"])
FACE_H = np.dot(P["chin"] - (EYE_L + EYE_R) / 2, -UP)
EYE_H = (np.dot(P["eye_l_bottom"] - P["eye_l_top"], -UP) + np.dot(P["eye_r_bottom"] - P["eye_r_top"], -UP)) / 2
MOUTH_W = math.dist(P["mouth_l"], P["mouth_r"])
if np.dot(P["mouth_lower"] - P["mouth_upper"], -UP) < 0.08 * MOUTH_W:   # 다문 입이면 아랫입술 점을 살짝 띄우기
    P["mouth_lower"] = P["mouth_upper"] - UP * 0.06 * MOUTH_W
sx, sy = np.mean([P["nose"], P["jaw_l"]], axis=0).astype(int)
SKIN = tuple(int(v) for v in char[sy, sx, :3])

# =============== 2. 메쉬 점 만들기 ===============
#   follow: 머리를 따라가는 정도 (1 = 머리, 0 = 고정)   depth: 고개 돌릴 때 움직이는 양 (코 > 눈 > 윤곽 > 머리카락)
WEIGHT = {"eye": (1, 1.0), "brow": (1, 1.0), "nose": (1, 1.15), "mouth": (1, 0.95),
          "face": (1, 0.35), "jaw": (1, 0.45), "chin": (1, 0.55), "neck": (0.25, 0.0)}
names, rest, follow, depth = [], [], [], []


def add(name, p, f, d):
    """메쉬 점 하나 추가 (자동 보조점이 다른 점과 너무 가까우면 건너뛰기)"""
    if name is None and rest and min(math.dist(p, q) for q in rest) < FACE_W * 0.04:
        return None
    names.append(name); rest.append(np.array(p, float)); follow.append(f); depth.append(d)
    return len(rest) - 1


for n, p in P.items():                                       # rig.json 의 점 27개
    add(n, p, *WEIGHT[n.split("_")[0]])
for c, rx, ry in ((EYE_L, math.dist(P["eye_l_a"], P["eye_l_b"]) * 0.68, EYE_H * 0.68),
                  (EYE_R, math.dist(P["eye_r_a"], P["eye_r_b"]) * 0.68, EYE_H * 0.68)):
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):  # 눈 둘레 보조점: 변형이 눈 주변에서만 일어나게
        add(None, c + RIGHT * math.cos(t) * rx - UP * math.sin(t) * ry, 1, 0.9)
LOW_RING = []                                                # 입 아래쪽 보조점 (턱과 같이 내려감)
MOUTH_C = (P["mouth_l"] + P["mouth_r"]) / 2
for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    i = add(None, MOUTH_C + RIGHT * math.cos(t) * MOUTH_W * 0.8 - UP * math.sin(t) * MOUTH_W * 0.45, 1, 0.85)
    if i is not None and math.sin(t) > 0.2:
        LOW_RING.append((i, math.sin(t)))
for e in rig.get("extra", []):                              # make_rig 에서 추가로 찍은 머리카락/몸 점
    add(None, (np.array(e["pos"]) + [PAD_X, PAD_Y]) * FIT, e["follow"], e["depth"])
for x in np.linspace(0, W - 1, 5):                           # 그림 가장자리: 고정
    for y in (0, H - 1):
        add("border", (x, y), 0, 0)
for y in np.linspace(0, H - 1, 5)[1:-1]:
    for x in (0, W - 1):
        add("border", (x, y), 0, 0)

REST, FOLLOW, DEPTH = np.array(rest), np.array(follow), np.array(depth)
I = {n: i for i, n in enumerate(names) if n}
PIVOT = REST[I["neck"]]                                      # 고개 기울기 회전 중심 = 목

# 들로네 삼각분할: 점들을 삼각형으로 잇기
sub = cv2.Subdiv2D((-1, -1, W + 2, H + 2))
for p in REST:
    sub.insert((float(p[0]), float(p[1])))
TRIS = []
for tri in sub.getTriangleList().reshape(-1, 3, 2):
    ids = [int(np.argmin(((REST - v) ** 2).sum(1))) for v in tri]
    if all(((REST[i] - v) ** 2).sum() < 1 for i, v in zip(ids, tri)):
        TRIS.append(ids)
print(f"메쉬: 점 {len(REST)}개, 삼각형 {len(TRIS)}개")


# =============== 3. 도우미 함수 ===============
smooth = {}


def soft(key, value, keep=0.6):
    """떨림 줄이기: 이전 값 60% + 새 값 40%"""
    smooth[key] = smooth.get(key, value) * keep + value * (1 - keep)
    return smooth[key]


def clip(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def curve(a, m, b, n=12):
    """a -> m -> b 를 부드럽게 지나는 곡선 위의 점 n 개"""
    t = np.linspace(0, 1, n)[:, None]
    c = 2 * m - (a + b) / 2
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def deform(tilt, zoom, mx, my, yaw, pitch, brow, mouth, smile):
    """숫자들 -> 메쉬 점들의 새 위치"""
    D = REST.copy()
    for n in ("brow_l_1", "brow_l_2", "brow_l_3", "brow_r_1", "brow_r_2", "brow_r_3"):
        D[I[n]] += UP * brow * EYE_H * 0.45                  # 눈썹 위아래
    G = mouth * MOUTH_W * 0.55                               # 입 벌어지는 높이
    D[I["mouth_upper"]] += UP * G * 0.12
    drop = -UP * G * 0.88                                    # 아랫입술이 내려가는 양
    for n, k in (("mouth_lower", 1), ("mouth_bottom", 1), ("chin", 0.55), ("jaw_l", 0.25), ("jaw_r", 0.25)):
        D[I[n]] += drop * k                                  # 턱은 덜, 턱선은 더 덜 내려감
    for i, k in LOW_RING:
        D[i] += drop * 0.7 * k
    side = RIGHT * (0.05 * smile - 0.06 * mouth) * MOUTH_W   # 미소: 입꼬리 바깥으로, 입 벌리면 안쪽으로
    D[I["mouth_l"]] += UP * smile * MOUTH_W * 0.1 - side
    D[I["mouth_r"]] += UP * smile * MOUTH_W * 0.1 + side
    D += (DEPTH * FOLLOW)[:, None] * [yaw * FACE_W * 0.14, pitch * FACE_H * 0.08]   # 고개 돌림 입체감
    a = np.radians(tilt) * FOLLOW                            # 목 기준 회전 (몸은 덜 / 안 따라감)
    v = D - PIVOT
    z = (1 + (zoom - 1) * FOLLOW)[:, None]
    D = PIVOT + np.stack([np.cos(a) * v[:, 0] - np.sin(a) * v[:, 1],
                          np.sin(a) * v[:, 0] + np.cos(a) * v[:, 1]], 1) * z + FOLLOW[:, None] * [mx, my]
    return D


def warp_triangle(src, dst, out):
    """원래 그림의 삼각형 src 를 새 위치 dst 로 늘여 붙이기"""
    x0, y0 = np.maximum(np.floor(dst.min(0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(dst.max(0)).astype(int) + 1, [W, H])
    if x1 <= x0 or y1 <= y0:
        return
    M = cv2.getAffineTransform(np.float32(src), np.float32(dst - [x0, y0]))
    piece = cv2.warpAffine(char, M, (int(x1 - x0), int(y1 - y0)), borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.round(dst - [x0, y0])), 255)
    cv2.copyTo(piece, mask, out[y0:y1, x0:x1])


# =============== 4. 실행 ===============
base = None                                                  # 기준 얼굴
show_mesh = False
while True:
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.time() * 1000))

    D = REST.copy()
    eye_open = {"l": 1.0, "r": 1.0}
    mouth = 0.0
    if result.face_landmarks:
        pts = [(p.x * w, p.y * h) for p in result.face_landmarks[0]]
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
        score = {c.category_name: c.score for c in result.face_blendshapes[0]}

        # ----- 내 얼굴 -> 숫자 -----
        eye_l = np.mean([pts[33], pts[133]], axis=0)
        eye_r = np.mean([pts[362], pts[263]], axis=0)
        mid, dist = (eye_l + eye_r) / 2, math.dist(eye_l, eye_r)
        now = {
            "angle": math.degrees(math.atan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])),      # 05번 기울기
            "yaw": (pts[1][0] - (pts[234][0] + pts[454][0]) / 2) / (abs(pts[454][0] - pts[234][0]) / 2),  # 코가 얼굴 가운데서 벗어난 정도
            "pitch": (pts[1][1] - mid[1]) / (pts[152][1] - mid[1]),                           # 눈~코 / 눈~턱
            "brow": score["browInnerUp"] * 0.5 + (score["browOuterUpLeft"] + score["browOuterUpRight"]) * 0.25
                    - (score["browDownLeft"] + score["browDownRight"]) * 0.5,
            "smile": (score["mouthSmileLeft"] + score["mouthSmileRight"]) / 2,
        }
        if base is None:                                     # 처음 얼굴을 기준으로 저장
            base = dict(now, mid=mid, dist=dist)
        gain = math.dist(EYE_L, EYE_R) / base["dist"] * 0.6
        D = deform(
            tilt=soft("tilt", clip(now["angle"] - base["angle"], -35, 35)),
            zoom=soft("zoom", clip(dist / base["dist"], 0.8, 1.25)),
            mx=soft("x", (mid[0] - base["mid"][0]) * gain),
            my=soft("y", (mid[1] - base["mid"][1]) * gain),
            yaw=soft("yaw", clip((now["yaw"] - base["yaw"]) * 2.2)),
            pitch=soft("pitch", clip((now["pitch"] - base["pitch"]) * 6)),
            brow=soft("brow", clip((now["brow"] - base["brow"]) * 3)),
            mouth=(mouth := soft("mouth", clip((math.dist(pts[13], pts[14]) / math.dist(pts[61], pts[291]) - 0.05) / 0.5, 0, 1), 0.3)),
            smile=soft("smile", clip((now["smile"] - base["smile"]) * 1.5)),
        )
        # 거울 화면이라 AI 의 'Right' 눈 = 화면 왼쪽(l) 눈
        eye_open = {"l": soft("eye_l", 1 - clip((score["eyeBlinkRight"] - 0.3) / 0.4, 0, 1), 0.3),
                    "r": soft("eye_r", 1 - clip((score["eyeBlinkLeft"] - 0.3) / 0.4, 0, 1), 0.3)}

    # ----- 메쉬 변형: 움직인 삼각형만 다시 붙이기 -----
    out = char.copy()
    moved = np.linalg.norm(D - REST, axis=1) > 0.2
    for tri in TRIS:
        if moved[tri].any():
            warp_triangle(REST[tri], D[tri], out)
    a = out[:, :, 3:] / 255
    canvas = (out[:, :, :3] * a + np.array(BG) * (1 - a)).astype(np.uint8)

    # ----- 눈꺼풀 덧그리기: 뜬 정도 o 만큼 위아래에서 덮어 오기 -----
    for side in ("l", "r"):
        o = eye_open[side]
        if o > 0.95:
            continue
        ea, top, eb, bottom = (D[I[f"eye_{side}_{k}"]] for k in ("a", "top", "b", "bottom"))
        c = (ea + eb) / 2 + (bottom - (ea + eb) / 2) * 0.35        # 위아래 눈꺼풀이 만나는 곳
        T, B = c + (top - c) * 1.3, c + (bottom - c) * 1.15        # 눈 전체 범위 (속눈썹까지)
        U, L = c + (T - c) * o, c + (B - c) * o                    # 지금 윗/아랫 눈꺼풀 위치
        keep = canvas.copy()
        cv2.fillPoly(canvas, [np.int32(np.vstack([curve(ea, T, eb), curve(eb, B, ea)]))], SKIN)
        hole = np.zeros(canvas.shape[:2], np.uint8)
        cv2.fillPoly(hole, [np.int32(np.vstack([curve(ea, U, eb), curve(eb, L, ea)]))], 255)
        canvas[hole > 0] = keep[hole > 0]                                                   # 뜬 부분은 원래대로
        cv2.polylines(canvas, [np.int32(curve(ea, U, eb))], False, (40, 30, 30), 3, cv2.LINE_AA)

    # ----- 입 안 덧그리기: 윗입술 곡선과 내려간 아랫입술 곡선 사이 -----
    if mouth > 0.05:
        ml, mr, up, low = (D[I[n]] for n in ("mouth_l", "mouth_r", "mouth_upper", "mouth_lower"))
        inside = np.int32(np.vstack([curve(ml, up, mr), curve(mr, low, ml)]))
        cv2.fillPoly(canvas, [inside], (50, 35, 100), cv2.LINE_AA)
        gap = np.linalg.norm(low - up)
        tc = low + (up - low) * 0.3
        angle = math.degrees(math.atan2((mr - ml)[1], (mr - ml)[0]))
        cv2.ellipse(canvas, (int(tc[0]), int(tc[1])), (int(MOUTH_W * 0.25), int(gap * 0.3)),
                    angle, 0, 360, (120, 110, 220), -1, cv2.LINE_AA)                     # 혀
        cv2.polylines(canvas, [inside], True, (60, 60, 170), 2, cv2.LINE_AA)

    if show_mesh:
        for tri in TRIS:
            cv2.polylines(canvas, [np.int32(D[tri])], True, (255, 200, 0), 1)

    cv2.imshow("camera", frame)
    cv2.imshow("vtuber", canvas)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("c"):
        base = None
    if key == ord("w"):
        show_mesh = not show_mesh
