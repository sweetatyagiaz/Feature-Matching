import cv2

from src.detectors.sift_detector import SIFTDetector
from src.matchers.sift_matcher import SIFTMatcher
from src.localization.homography import find_object


query = cv2.imread(
    "data/query/query.jpg",
    cv2.IMREAD_GRAYSCALE
)

target = cv2.imread(
    "data/target/target.jpg",
    cv2.IMREAD_GRAYSCALE
)

detector = SIFTDetector()

kp1, des1 = detector.extract(query)
kp2, des2 = detector.extract(target)

matcher = SIFTMatcher()

matches = matcher.match(
    des1,
    des2
)

print(f"Good Matches: {len(matches)}")

polygon = find_object(
    query,
    target,
    kp1,
    kp2,
    matches
)

target_color = cv2.cvtColor(
    target,
    cv2.COLOR_GRAY2BGR
)

if polygon is not None:

    cv2.polylines(
        target_color,
        [polygon.astype(int)],
        True,
        (0, 255, 0),
        5
    )

cv2.imwrite(
    "data/output/result.jpg",
    target_color
)