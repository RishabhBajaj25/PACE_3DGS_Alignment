#pip install defisheye
from defisheye import Defisheye
import cv2
import numpy as np

dtype = 'equalarea'
format = 'fullframe'
fov = 180
pfov = 90

img = "/home/rishabh/datasets/instax5/calibration/screen_side/INSV/frame_0001.jpg"
cv2_img = cv2.imread(img)

img_dim = cv2_img.shape[0]
R = img_dim / 2

# ---- Compute maximum safe shift ----
r_pfov = R * (pfov / fov)
max_shift = R - r_pfov

overlap_factor = 0.6
shift = 200  # your manual override

xcenters = [R - shift, R, R + shift, R]
ycenters = [R - shift, R, R + shift, R]

for xcenter in xcenters:
    for ycenter in ycenters:

        obj = Defisheye(
            img,
            dtype=dtype,
            format=format,
            fov=fov,
            pfov=pfov,
            xcenter=xcenter,
            ycenter=ycenter
        )

        new_image = obj.convert()

        # ---- Add text bottom-left ----
        text = f"x: {int(xcenter)}, y: {int(ycenter)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2

        h, w = new_image.shape[:2]
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        text_x = 10
        text_y = h - 10  # bottom margin

        # Optional: draw background rectangle for readability
        cv2.rectangle(
            new_image,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            new_image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        img_out = f"/home/rishabh/datasets/instax5/calibration/screen_side/INSV/frame_0001_{int(xcenter)}_{int(ycenter)}.jpg"
        cv2.imwrite(img_out, new_image)