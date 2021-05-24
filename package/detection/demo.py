import sys
import centerface
from PIL import Image

if len(sys.argv) != 2:
    print("Usage: python demo.py path/to/img")
else:
    img = sys.argv[1]
    img = Image.open(img)
    res = centerface.detect([img])
    res = res[0]
    centerface.visualize(img, res['bbox'], res['landmarks'])