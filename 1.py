import numpy as np
from PIL import Image

m = np.array(Image.open("labels.png"), dtype=np.uint8)   # 0/1
Image.fromarray(m * 255).save("labels_vis.png")
