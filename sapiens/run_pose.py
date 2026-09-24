# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch>=2.4", "opencv-python>=4.9", "onnxruntime>=1.18", "numpy"]
# ///
"""Sapiens (Meta) 308-keypoint pose on a video or camera, for the closest person.

Detector: YOLOX-tiny (HumanArt, ONNX). Pose: Sapiens-Pose TorchScript (1024x768 input).

Usage:
    uv run sapiens/run_pose.py --input swing.mov
    uv run sapiens/run_pose.py --input swing.mov --model 1b --out swing_pose.mp4
    uv run sapiens/run_pose.py --input 0            # camera (slow, not real-time)
Outputs an annotated video and an .npz with keypoints (frames x 308 x 3: x, y, score).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from classes_and_palettes import GOLIATH_KPTS_COLORS, GOLIATH_SKELETON_INFO
from pose_utils import udp_decode

HERE = Path(__file__).parent
CKPT = HERE / "checkpoints"
MODELS = {
    "0.3b": ("facebook/sapiens-pose-0.3b-torchscript", "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"),
    "0.6b": ("facebook/sapiens-pose-0.6b-torchscript", "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"),
    "1b": ("facebook/sapiens-pose-1b-torchscript", "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"),
}
POSE_W, POSE_H, HEATMAP_SCALE = 768, 1024, 4
MEAN = np.array([123.5, 116.5, 103.5], np.float32)  # RGB
STD = np.array([58.5, 57.0, 57.5], np.float32)
DET_SIZE, DET_THR = 416, 0.4
LINKS = [(v["link"], tuple(reversed(v["color"]))) for v in GOLIATH_SKELETON_INFO.values()]  # RGB -> BGR
LINKED = sorted({i for (a, b), _ in LINKS for i in (a, b)})


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_path(size: str) -> Path:
    repo, fname = MODELS[size]
    p = CKPT / f"sapiens_{size}_pose.pt2"
    if not p.exists():
        raise SystemExit(
            f"missing {p}\n  curl -L -o {p} https://huggingface.co/{repo}/resolve/main/{fname}")
    return p


def detect_closest(det: ort.InferenceSession, frame: np.ndarray):
    """Largest person box [x1, y1, x2, y2] (largest = closest to the camera), or None."""
    h, w = frame.shape[:2]
    r = min(DET_SIZE / w, DET_SIZE / h)
    x = np.full((DET_SIZE, DET_SIZE, 3), 114, np.float32)
    x[: round(h * r), : round(w * r)] = cv2.resize(frame, (round(w * r), round(h * r)))  # BGR
    dets, labels = det.run(None, {"input": x.transpose(2, 0, 1)[None]})
    d = dets[0][(dets[0][:, 4] > DET_THR) & (labels[0] == 0)]
    if not len(d):
        return None
    d[:, :4] /= r
    return max(d[:, :4], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def crop_params(box):
    """Center and size of the 3:4 crop around a box, padded 1.25x (mmpose TopdownAffine)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = (box[2] - box[0]) * 1.25, (box[3] - box[1]) * 1.25
    if bw > bh * POSE_W / POSE_H:
        bh = bw * POSE_H / POSE_W
    else:
        bw = bh * POSE_W / POSE_H
    return np.array([cx, cy]), np.array([bw, bh])


@torch.inference_mode()
def estimate(model, device, dtype, frame, box):
    center, size = crop_params(box)
    s = POSE_W / size[0]
    m = np.array([[s, 0, POSE_W / 2 - s * center[0]], [0, s, POSE_H / 2 - s * center[1]]], np.float32)
    crop = cv2.warpAffine(frame, m, (POSE_W, POSE_H), flags=cv2.INTER_LINEAR)
    x = (crop[..., ::-1].astype(np.float32) - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(device, dtype)
    heatmaps = model(t)[0].float().cpu().numpy()
    kps, scores = udp_decode(heatmaps, (POSE_W, POSE_H),
                             (POSE_W // HEATMAP_SCALE, POSE_H // HEATMAP_SCALE))
    kps = kps[0] / [POSE_W, POSE_H] * size + center - size / 2
    return np.concatenate([kps, scores[0][:, None]], axis=1)  # (308, 3)


def draw(frame, kps, thr):
    # Fingers (21+) get a stricter threshold: when a hand is hidden, low-confidence finger
    # points can land on other body parts and draw long stray lines.
    thr = np.where(np.arange(len(kps)) < 21, thr, max(thr, 0.5))
    lw = max(2, frame.shape[1] // 400)
    for (a, b), color in LINKS:
        if kps[a, 2] >= thr[a] and kps[b, 2] >= thr[b]:
            cv2.line(frame, tuple(map(int, kps[a, :2])), tuple(map(int, kps[b, :2])),
                     color, lw, cv2.LINE_AA)
    for i in LINKED:
        if kps[i, 2] >= thr[i]:
            c = tuple(int(v) for v in reversed(GOLIATH_KPTS_COLORS[i]))
            cv2.circle(frame, tuple(map(int, kps[i, :2])), lw + 1, c, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video path or camera index")
    ap.add_argument("--model", default="0.3b", choices=MODELS)
    ap.add_argument("--out", default=None, help="output video (default: <input>_sapiens.mp4)")
    ap.add_argument("--kpt-thr", type=float, default=0.3)
    ap.add_argument("--device", default="auto", help="auto / mps / cuda / cpu")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--no-show", action="store_true", help="don't open a preview window")
    ap.add_argument("--fp32", action="store_true", help="full precision (slower on mps/cuda)")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(device.type, torch.float32)
    if args.fp32:
        dtype = torch.float32
    print(f"device={device} dtype={dtype} model={args.model}")
    model = torch.jit.load(str(model_path(args.model)), map_location="cpu").eval().to(device, dtype)
    det = ort.InferenceSession(str(CKPT / "yolox-tiny-humanart.onnx"), providers=["CPUExecutionProvider"])

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stem = "camera" if isinstance(src, int) else Path(args.input).with_suffix("").as_posix()
    out_path = args.out or f"{stem}_sapiens.mp4"

    writer, all_kps, times, box = None, [], [], None
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and len(all_kps) >= args.max_frames):
            break
        t0 = time.perf_counter()
        # Re-detect every 15 frames or when lost; otherwise follow the previous keypoints.
        if box is None or len(all_kps) % 15 == 0:
            box = detect_closest(det, frame)
        if box is None:
            kps = np.zeros((308, 3), np.float32)
        else:
            kps = estimate(model, device, dtype, frame, box)
            body = kps[:15][kps[:15, 2] >= args.kpt_thr]  # nose..ankles
            box = None if len(body) < 4 else [*body[:, :2].min(0), *body[:, :2].max(0)]
            if box is not None:  # body joints miss the head top and soles; widen
                c = (np.array(box[:2]) + box[2:]) / 2
                half = (np.array(box[2:]) - box[:2]) / 2 * [1.15, 1.2]
                box = [*(c - half), *(c + half)]
        times.append(time.perf_counter() - t0)
        all_kps.append(kps)

        draw(frame, kps, args.kpt_thr)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        if len(all_kps) % 10 == 0:
            print(f"frame {len(all_kps)}: {1000 * np.mean(times[-10:]):.0f} ms/frame")
        if not args.no_show:
            cv2.imshow("sapiens", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    np.savez_compressed(Path(out_path).with_suffix(".npz"), keypoints=np.array(all_kps), fps=fps)
    if times:
        print(f"done: {len(times)} frames, avg {1000 * np.mean(times):.0f} ms/frame -> {out_path}")


if __name__ == "__main__":
    main()
