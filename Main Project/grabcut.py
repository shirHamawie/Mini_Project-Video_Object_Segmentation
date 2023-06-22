import cv2
import numpy as np
from typing import Tuple


class GrabCut:
    def __init__(self, image: np.ndarray, complement_with_white=False):
        self.image = image
        self.mask = np.zeros(self.image.shape[:2], np.uint8)
        self.window_name = 'Mark Foreground'
        self.foreground = None
        self.background = None
        self.complement_with_white = complement_with_white

    def _perform_grabcut(self, rect) -> Tuple[np.ndarray, np.ndarray]:
        foreground = np.zeros((1, 65), np.float64)
        background = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            img=self.image,
            mask=self.mask,
            rect=rect,
            bgdModel=background,
            fgdModel=foreground,
            iterCount=10,
            mode=cv2.GC_INIT_WITH_RECT
        )

        # Assign definite background and probable background as 0, others as 1
        mask = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        inverted_mask = 1 - mask

        # Apply the mask to the input image as the foreground
        foreground = self.image * mask[:, :, np.newaxis]

        # Apply the inverted mask to the input image as the background
        background = self.image * inverted_mask[:, :, np.newaxis]

        if self.complement_with_white:
            foreground = foreground + 255 * inverted_mask[:, :, np.newaxis]
            background = background + 255 * mask[:, :, np.newaxis]

        return foreground, background

    def run(self, show=True):
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, self.image)
        object_box = cv2.selectROI(self.window_name, self.image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(self.window_name)

        foreground, background = self._perform_grabcut(object_box)

        if show:
            cv2.imshow('Foreground', foreground)
            cv2.imshow('Background', background)
            cv2.waitKey(0)

        self.foreground = foreground
        self.background = background

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # test grabcut
    image = cv2.imread('data/images/tinky_winky.jpeg')
    grabcut = GrabCut(image)
    grabcut.run()
    cv2.imshow('Foreground', grabcut.foreground)
    cv2.imshow('Background', grabcut.background)
    cv2.imwrite('data/images/tinky_winky_foreground.jpg', grabcut.foreground)
    cv2.imwrite('data/images/tinky_winky_background.jpg', grabcut.background)

