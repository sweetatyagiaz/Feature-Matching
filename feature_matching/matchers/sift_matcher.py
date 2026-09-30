import cv2


class SIFTMatcher:

    def __init__(self):

        index_params = dict(
            algorithm=1,
            trees=5
        )

        search_params = dict(
            checks=50
        )

        self.matcher = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )

    def match(self, desc1, desc2):

        matches = self.matcher.knnMatch(
            desc1,
            desc2,
            k=2
        )

        good_matches = []

        for m, n in matches:

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches