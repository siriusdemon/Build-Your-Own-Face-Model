import os
import os.path as osp
import sys
import centerface
from PIL import Image

if len(sys.argv) != 2:
    print("Usage: python demo.py path/to/img(or dir)")
else:
    if osp.isfile(sys.argv[1]):
        img = Image.open(sys.argv[1])
        img = [img]
    elif osp.isdir(sys.argv[1]):
        files = os.listdir(sys.argv[1])
        img = [Image.open(osp.join(sys.argv[1], f)) for f in files]
    else:
        os._exit(1)
    res = centerface.detect(img)
    for (r, im) in zip(res, img):
        centerface.visualize(im, r['bbox'], r['landmarks'])