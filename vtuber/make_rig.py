"""내 캐릭터 그림에 점 27개를 찍어서 rig.json 만들기 (이 파일은 따라 치지 않고 그대로 씁니다)

  python make_rig.py my_char/character.png      ->  my_char/rig.json 이 만들어져요

- 터미널과 창 왼쪽 위에 '다음에 찍을 점' 이 나옵니다. 왼쪽/오른쪽은 항상 '화면 기준' 이에요.
- 27개를 다 찍은 뒤:  좌클릭 = 머리카락 점 추가,  우클릭 = 몸(고정) 점 추가,  점 드래그 = 위치 수정
- 키 (그림 창을 클릭한 상태에서):  u 되돌리기   s 저장   q 종료
- 만들어진 rig.json 은 09번 코드와 완성 코드(main.py) 에서 모두 쓸 수 있어요.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REQUIRED_POINTS = [
    ('eye_l_a', '화면 왼쪽 눈 - 왼쪽 끝'),
    ('eye_l_top', '화면 왼쪽 눈 - 윗눈꺼풀 가운데'),
    ('eye_l_b', '화면 왼쪽 눈 - 오른쪽 끝'),
    ('eye_l_bottom', '화면 왼쪽 눈 - 아랫눈꺼풀 가운데'),
    ('eye_r_a', '화면 오른쪽 눈 - 왼쪽 끝'),
    ('eye_r_top', '화면 오른쪽 눈 - 윗눈꺼풀 가운데'),
    ('eye_r_b', '화면 오른쪽 눈 - 오른쪽 끝'),
    ('eye_r_bottom', '화면 오른쪽 눈 - 아랫눈꺼풀 가운데'),
    ('brow_l_1', '왼쪽 눈썹 - 왼쪽'),
    ('brow_l_2', '왼쪽 눈썹 - 가운데'),
    ('brow_l_3', '왼쪽 눈썹 - 오른쪽'),
    ('brow_r_1', '오른쪽 눈썹 - 왼쪽'),
    ('brow_r_2', '오른쪽 눈썹 - 가운데'),
    ('brow_r_3', '오른쪽 눈썹 - 오른쪽'),
    ('nose', '코끝'),
    ('mouth_l', '입 - 왼쪽 끝'),
    ('mouth_r', '입 - 오른쪽 끝'),
    ('mouth_top', '윗입술 위쪽 (입 위 피부 경계)'),
    ('mouth_upper', '윗입술 안쪽 (입이 닫혀 있으면 입술선)'),
    ('mouth_lower', '아랫입술 안쪽 (입이 닫혀 있으면 입술선과 같은 위치)'),
    ('mouth_bottom', '아랫입술 아래쪽'),
    ('face_l', '얼굴 윤곽 - 왼쪽 볼'),
    ('jaw_l', '얼굴 윤곽 - 왼쪽 턱선'),
    ('chin', '턱 끝'),
    ('jaw_r', '얼굴 윤곽 - 오른쪽 턱선'),
    ('face_r', '얼굴 윤곽 - 오른쪽 볼'),
    ('neck', '목 (머리 회전 중심)'),
]

WIN = "annotate"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("사용법: python make_rig.py 그림파일.png")
        return
    img_path = Path(sys.argv[1]).resolve()
    rig_path = img_path.parent / "rig.json"
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {img_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # show on checker-free light background
        a = img[:, :, 3:] / 255.0
        img = (img[:, :, :3] * a + 230 * (1 - a)).astype(np.uint8)
    view_scale = min(1.0, 900 / max(img.shape[:2]))

    points, extra = {}, []
    if rig_path.exists():
        cfg = json.loads(rig_path.read_text())
        points = {k: list(map(float, v)) for k, v in cfg.get("points", {}).items()}
        extra = cfg.get("extra", [])
        print(f"loaded existing {rig_path}")
    history = []
    drag = {"target": None}

    def next_required():
        for name, desc in REQUIRED_POINTS:
            if name not in points:
                return name, desc
        return None

    def all_handles():
        hs = [("p", k, v) for k, v in points.items()]
        hs += [("e", i, e["pos"]) for i, e in enumerate(extra)]
        return hs

    def nearest(x, y, r=10 / view_scale):
        best, bd = None, r
        for h in all_handles():
            d = np.hypot(h[2][0] - x, h[2][1] - y)
            if d < bd:
                best, bd = h, d
        return best

    def on_mouse(ev, x, y, flags, _):
        x, y = x / view_scale, y / view_scale
        if ev == cv2.EVENT_LBUTTONDOWN:
            nxt = next_required()
            if nxt:  # 필수 점을 찍는 동안에는 클릭 = 항상 다음 점 찍기 (같은 자리에 겹쳐 찍을 수 있게)
                points[nxt[0]] = [x, y]
                history.append(("p", nxt[0]))
                return
            h = nearest(x, y)
            if h:    # 다 찍은 뒤에는 점 근처를 누르면 드래그로 위치 수정
                drag["target"] = h[:2]
                return
            else:
                extra.append({"pos": [x, y], "follow": 1.0, "depth": 0.15})
                history.append(("e", len(extra) - 1))
        elif ev == cv2.EVENT_RBUTTONDOWN and not next_required():
            extra.append({"pos": [x, y], "follow": 0.0, "depth": 0.0})
            history.append(("e", len(extra) - 1))
        elif ev == cv2.EVENT_MOUSEMOVE and drag["target"]:
            kind, key = drag["target"]
            if kind == "p":
                points[key] = [x, y]
            else:
                extra[key]["pos"] = [x, y]
        elif ev == cv2.EVENT_LBUTTONUP:
            drag["target"] = None

    def save():
        missing = [n for n, _ in REQUIRED_POINTS if n not in points]
        if missing:
            print(f"still missing: {missing}")
            return False
        cfg = {"image": img_path.name, "points": {k: [round(v[0], 1), round(v[1], 1)] for k, v in points.items()},
               "extra": extra, "settings": {}}
        if rig_path.exists():
            cfg["settings"] = json.loads(rig_path.read_text()).get("settings", {})
        rig_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"saved {rig_path}")
        return True

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    last_prompt = None
    while True:
        canvas = img.copy()
        for k, v in points.items():
            p = tuple(int(c) for c in v)
            cv2.circle(canvas, p, 4, (0, 0, 255), -1)
            cv2.putText(canvas, k, (p[0] + 5, p[1] - 4), 0, 0.35, (0, 0, 180), 1)
        for e in extra:
            p = tuple(int(c) for c in e["pos"])
            cv2.circle(canvas, p, 4, (255, 120, 0) if e["follow"] > 0 else (120, 120, 120), -1)
        nxt = next_required()
        prompt = f"click: {nxt[0]}  ({nxt[1]})" if nxt else \
            "done! L-click: head pt, R-click: body pt, drag to edit, s: save"
        if prompt != last_prompt:
            print(prompt)
            last_prompt = prompt
        canvas = cv2.resize(canvas, None, fx=view_scale, fy=view_scale)
        cv2.putText(canvas, prompt.split("  (")[0], (8, 22), 0, 0.6, (0, 0, 0), 3)
        cv2.putText(canvas, prompt.split("  (")[0], (8, 22), 0, 0.6, (255, 255, 255), 1)
        cv2.imshow(WIN, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("u") and history:
            kind, key = history.pop()
            if kind == "p":
                points.pop(key, None)
            else:
                extra.pop(key)
        if k == ord("s"):
            save()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
