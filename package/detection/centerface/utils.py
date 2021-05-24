import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw


class VisionKit:

    @staticmethod
    def letterbox(im, size, bboxes=None, landmarks=None, skip=3):
        """Scale im to target size while keeping its w/h ratio
        Args:
            im: PIL.Image object
            size: target size, tuple or list
            bboxes: array([n, left, top, right, bottom])
            landmarks: array([n, x, y, score])
            skip: NOTE that in retinaface annotations file, each landmark includes 
                a score, if you want to use this function in other task, remember 
                to reset it to properly value, e.g. skip=2.
        Returns:
            im, bboxes, landmarks, scale, offset_x, offset_y
        """
        canvas = Image.new("RGB", size=size, color="#777")
        target_width, target_height = size
        width, height = im.size
        offset_x = 0
        offset_y = 0
        if height > width:
            height_ = target_height
            scale = height_ / height
            width_ = int(width * scale)  # make sure h_ / w_ == h / w
            offset_x = (target_width - width_) // 2
        else:
            width_ = target_width
            scale = width_ / width
            height_ = int(height * scale)
            offset_y = (target_height - height_) // 2
        im = im.resize((width_, height_), Image.BILINEAR)
        canvas.paste(im, box=(offset_x, offset_y))

        if bboxes is not None:
            bboxes = bboxes.copy()
            bboxes *= scale
            bboxes[:, 0::2] += offset_x
            bboxes[:, 1::2] += offset_y        

        if landmarks is not None:
            landmarks = landmarks.copy()
            landmarks *= scale
            landmarks[:, 0::skip] += offset_x
            landmarks[:, 1::skip] += offset_y

        return canvas, bboxes, landmarks, scale, offset_x, offset_y

    @staticmethod
    def letterbox_inverse(scale, offset_x, offset_y, bboxes=None, landmarks=None, skip=3):
        if bboxes is not None:
            bboxes = np.array(bboxes)
            bboxes[:, 0::2] -= offset_x
            bboxes[:, 1::2] -= offset_y
            bboxes /= scale
        if landmarks is not None:
            landmarks = np.array(landmarks)
            landmarks[:, 0::skip] -= offset_x
            landmarks[:, 1::skip] -= offset_y
            landmarks /= scale
        return bboxes, landmarks


    @staticmethod
    def visualize(im, bboxes=[], landmarks=[], skip=3):
        im = im.copy()
        handle = ImageDraw.Draw(im)
        for bbox in bboxes:
            # draw bbox
            left, top, right, bottom = map(int, bbox)
            handle.rectangle([left, top, right, bottom], outline=(0,0,255), width=1)
            # draw center Point
            width = right - left
            height = bottom - top
            center_x = left + width // 2
            center_y = top + height // 2
            handle.ellipse([center_x-1, center_y-1, center_x+1, center_y+1], width=1)

        for landmark in landmarks:
            for i in range(0, len(landmark), skip):
                x, y = int(landmark[i]), int(landmark[i+1])
                handle.ellipse([x-1, y-1, x+1, y+1], fill=(0,127,0), width=1)
        im.show()
        return im

    @staticmethod
    def nms(heat, kernel):
        padding = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, kernel, stride=1, padding=padding)
        keep = (hmax == heat).float()
        return heat * keep