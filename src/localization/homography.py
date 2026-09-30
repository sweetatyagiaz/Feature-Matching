import cv2
import numpy as np


def find_object(
    query_image,
    target_image,
    kp1,
    kp2,
    matches
):

    if len(matches) < 10:
        return None

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        return None

    h, w = query_image.shape[:2]

    corners = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(
        corners,
        H
    )

    return projected