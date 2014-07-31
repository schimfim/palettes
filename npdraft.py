import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt
import matplotlib.cm as cm

# input image
img = Image.open('orig/kueche.jpg')
img.thumbnail((256,256))
ary = np.asarray(img)/255.0

# match colors
match = np.array([[1.0,0,0],
               [0,1.0,0],
               [0,0,1.0]])
#match = np.array([[0,0,1.0]])

# max brightness
hsv = rgb_to_hsv(ary)
hsv[:,:,2] = 1.0
flat = hsv_to_rgb(hsv)

# main loop
res = np.zeros_like(flat)
res_map = np.zeros_like(flat)
#res = np.copy(ary)
for col in match:
	print col
	
	# calc distance
	diff = 1 - np.absolute(flat - col)
	k = np.array(np.ones((3,3))) / 3.0
	diff = np.inner(diff, k)
	diff = np.square(diff)

	ths = 0.4
	map = diff - ths
	map = np.sign(map) / 2 + 0.5
	map = map * diff
	plt.imshow(map[:,:,0], cmap=cm.jet)
	plt.show()

	res_map += map
	res += map * col

# output image
res_map = np.clip(res_map, 0,1)
out = ary * (1-res_map) + res
out = np.clip(out, 0,1)
out *= 255
nimg = Image.fromarray(out.astype(np.uint8))

nimg.show()

