import sys
from typing import Sequence

import time
import cv2
import numpy as np


def edge_filter(gray: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(
        gray, cv2.CV_32F, 1, 0, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_DEFAULT
    )
    grad_y = cv2.Sobel(
        gray, cv2.CV_32F, 0, 1, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_DEFAULT
    )
    grad_norm = np.sqrt(np.square(grad_x) + np.square(grad_y))
    return np.clip(grad_norm, 0.0, 255.0).astype(np.uint8)
    return (grad_norm > 100.0).astype(np.uint8) * 255


def draw_flow_arrows(
    img: np.ndarray, flow: np.ndarray, spacing: int, scale: float, color: Sequence[float]
) -> None:
    assert img.shape[0] == flow.shape[0] and img.shape[1] == flow.shape[1]

    height, width = img.shape[:2]
    y, x = np.mgrid[spacing // 2 : height : spacing, spacing // 2 : width : spacing].astype(
        np.int32
    )
    f = cv2.resize(flow, dsize=x.shape[::-1], interpolation=cv2.INTER_AREA)

    # Fixed-point for sub-pixel positioning
    shift = 4
    x <<= shift
    y <<= shift
    f = (f * 2**shift * scale).astype(np.int32)

    for start_x, start_y, (dx, dy) in zip(x.flatten(), y.flatten(), f.reshape((-1, 2))):
        cv2.arrowedLine(
            img,
            (start_x, start_y),
            (start_x + dx, start_y + dy),
            color=color,
            thickness=1,
            tipLength=0.3,
            shift=shift,
            line_type=cv2.LINE_AA,
        )


def main(file_path: str) -> None:

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Failed to open video '{file_path}'")
        exit()

    framerate = cap.get(cv2.CAP_PROP_FPS)
    if framerate == 0.0:
        print(f"Using video framerate of {framerate} fps")
    else:
        framerate = 30.0
        print(f"Unable to determine video framerate, using {framerate} fps")

    imgs = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        imgs.append(frame)
    cap.release()

    img = np.zeros(shape=(imgs[0].shape[0] * 2, imgs[0].shape[1] * 2, 3), dtype=np.uint8)

    prev = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)

    hsv = np.zeros_like(imgs[0])
    hsv[..., 1] = 255

    last_time = time.time()

    orb = cv2.ORB_create()

    i = 1
    while True:
        next = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        edges_next = edge_filter(next)

        flow = cv2.calcOpticalFlowFarneback(
            prev=prev,
            next=next,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=10,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = np.rad2deg(ang) / 2
        hsv[..., 2] = cv2.normalize(src=mag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        img_top_left = img[: img.shape[0] // 2, : img.shape[1] // 2, :]
        img_bottom_left = img[img.shape[0] // 2 :, : img.shape[1] // 2, :]
        img_top_right = img[: img.shape[0] // 2, img.shape[1] // 2 :, :]

        img_top_left[:] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
        draw_flow_arrows(
            img=img_top_left,
            flow=flow,
            spacing=10,
            scale=5,
            color=(255, 255, 255),
        )
        img_bottom_left[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_top_right[:] = cv2.cvtColor(edges_next, cv2.COLOR_GRAY2BGR)

        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        cv2.drawKeypoints(
            img_top_right,
            keypoints=kp,
            outImage=img_top_right,
            color=(0, 255, 0),
            flags=0,
        )

        cv2.imshow("img", img)

        current_time = time.time()
        elapsed = current_time - last_time
        time_to_wait = max(int(1000.0 / framerate - 1000 * elapsed), 1)
        last_time = current_time

        key = cv2.waitKey(time_to_wait)
        if key == 27:  # Esc
            break

        prev = next
        i = (i + 1) % len(imgs)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        print("Specify a video file to open and process it")
