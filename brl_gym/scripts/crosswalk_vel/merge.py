from PIL import Image

import numpy as np

import glob

files = glob.glob("imgs/*.png")
files.sort()

print('\n'.join(files))

bg = Image.open(files[0])

#bg = Image.new("RGBA", bg.size)
for i in range(0, len(files)):
    img = Image.open(files[i])

    alpha = int(120+128 * ((len(files)-i-1)/len(files)))
    print(alpha)
    #img.putalpha(alpha)
    #bg = Image.alpha_composite(bg, img)
    bg.paste(img, (0,0), img)
bg.save('merged.png', 'PNG')




