import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

mapfile = os.path.join(dir_path, "../assets/walls.png")
img = Image.open(mapfile)
arr = np.array(img)

# plt.imshow(arr)
# plt.show()


import IPython; IPython.embed()

