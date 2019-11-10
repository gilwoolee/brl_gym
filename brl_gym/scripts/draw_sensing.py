from PIL import Image, ImageDraw
import numpy as  np
def convert_coordinates_to_img_coords(
    im_size, xy, xlim=[-1.5,1.5], ylim=[-1.5,1.5]):
    im_w, im_h = im_size
    x, y = xy[:, 0], xy[:, 1]

    # flip y
    y = y * - 1
    x = im_w * (x - xlim[0]) / (xlim[1] - xlim[0])
    y = im_h * (y - ylim[0]) / (ylim[1] - ylim[0])

    return np.around(np.array([x, y]).transpose())

im = Image.open("resource/maze10_green.png")
draw = ImageDraw.Draw(im)

size = im.size


coords = np.genfromtxt("maze10_sensing.txt", delimiter="\t")

xy = convert_coordinates_to_img_coords(im.size, coords)
r = 1
for x,y in xy:
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,255))


# draw.line((0, 0) + im.size, fill=128)
# draw.line((0, im.size[1], im.size[0], 0), fill=128)

del draw

im.save("maze10_sensing_location.png", "PNG")


