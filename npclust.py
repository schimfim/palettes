import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt

nhist = 8
minh = 2.0/nhist
nfilt = 12
min_sat = 0.5 

def gen_hsv(n):
	hi = np.outer(np.arange(0.0, 1.0, 1.0/n), np.ones(n))
	si = hi.T
	mat = np.ones((n,n,3))
	mat[:,:,0] = hi
	mat[:,:,1] = si
	return mat

# input image
img = Image.open('orig/pond.jpg')
img.thumbnail((256,256))
ary = np.asarray(img)/255.0

# get hue and saturation
hsv = rgb_to_hsv(ary)
hues = hsv[:,:,0].flatten()
sats = hsv[:,:,1].flatten()

# remove low saturation
hues = hues[hsv[:,:,1].flatten() >= min_sat]
sats = sats[hsv[:,:,1].flatten() >= min_sat]

hist = np.histogram2d(hues, sats, nhist, normed=False )
vals = np.sqrt(hist[0]/np.max(hist[0]))
mat = gen_hsv(nhist)
mat[:,:,2] = vals
rgb = hsv_to_rgb(mat)

# show histogram
plt.imshow(rgb, interpolation='none')
plt.show()
plt.clf()

mat[mat[:,:,2]<minh] = 0
mat = mat[mat[:,:,2]>0]
l = mat.shape
nc = l[0]
mat = mat.reshape(1,nc,3)

# show proto colors
c = hsv_to_rgb(mat)
plt.imshow(c, interpolation='none')
plt.show()
plt.clf()

filt = gen_hsv(nfilt)
l = filt.shape
filt = filt.reshape(1,l[0]*l[1],3)
'''
c = hsv_to_rgb(filt)
plt.imshow(c, interpolation='none')
plt.show()
plt.clf()
'''
# calc dist
mhue = np.outer(mat[:,:,0], np.ones(nfilt**2))
msat = np.outer(mat[:,:,1], np.ones(nfilt**2))
fhue = np.outer(np.ones(nc), filt[:,:,0])
fsat = np.outer(np.ones(nc), filt[:,:,1])

dhue = np.square(mhue-fhue)
dsat = np.square(msat-fsat)
dist = np.sqrt(dhue+dsat)
mind = np.argmin(dist, 0)

# colors at minimum distance
# mat and filt are now 1xMx3
chue = mat[0,mind,0]
csat = mat[0,mind,1]
cval = mat[0,mind,2]
cfilt = np.copy(filt)
cfilt[0,:,0] = chue
cfilt[0,:,1] = csat
cfilt[0,:,2] = cval
mfilt = cfilt.reshape(nfilt, nfilt, 3)

#
c = hsv_to_rgb(mfilt)
plt.imshow(c, interpolation='none')
plt.show()
plt.clf()

# test image
img = Image.open('orig/dunes.jpg')
img.thumbnail((256,256))
ary = np.asarray(img)/255.0
hsv = rgb_to_hsv(ary)

hsv_filt = np.copy(hsv)
idx = (hsv*nfilt). astype(np.int)
ff = cfilt.reshape(nfilt*nfilt,3)
hue_filt = ff[idx[:,:,0]]
sat_filt = ff[idx[:,:,1]]
val_filt = ff[idx[:,:,2]]
val_filt = hsv
out = np.dstack((hue_filt[:,:,0], sat_filt[:,:,1], val_filt[:,:,2]))

c = hsv_to_rgb(out)
plt.imshow(c, interpolation='none')
plt.show()
plt.clf()

