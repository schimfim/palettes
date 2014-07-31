import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt

# input image
img = Image.open('orig/herbst.jpg')
img.thumbnail((256,256))
ary = np.asarray(img)/255.0

# get hue and saturation
hsv = rgb_to_hsv(ary)
hues = hsv[:,:,0].flatten()
sats = hsv[:,:,1].flatten()

hist = np.histogram2d(hues, sats, 16, normed=False  )
hi = np.outer(np.linspace(0.0, 1.0, 16), np.ones(16))
si = hi.T
mat = np.zeros((16,16,3))
mat[:,:,0] = hi
mat[:,:,1] = si
mat[:,:,2] = np.sqrt(hist[0]/np.max(hist[0]))
rgb = hsv_to_rgb(mat)

plt.imshow(rgb, interpolation='none')
plt.show()
