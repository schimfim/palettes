import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt

N = 8
minh = 0.3

# input image
img = Image.open('orig/pond.jpg')
img.thumbnail((256,256))
ary = np.asarray(img)/255.0

# get hue and saturation
hsv = rgb_to_hsv(ary)
hsv = np.reshape(hsv, (-1,3))

hist, edges = np.histogramdd(hsv, bins=(N,N,N), range=((0,1), (0,1), (0,1)), normed=True )

# full hsv meshes NxNxN
ax = np.linspace(0, 1, N)
mh,ms,mv = np.meshgrid(ax,ax,ax)

# show pure cube
N3 = N**3
pure = np.dstack((mh.reshape(N3), ms.reshape(N3), mv.reshape(N3))). squeeze() . reshape(N*2,N*N/2,3)

plt.imshow(hsv_to_rgb(pure))
plt.show()
plt.clf()

# reduces hsv meshes LxLxL
rh = mh[hist>=minh]
rs = ms[hist>=minh]
rv = mv[hist>=minh]
l = (rh.shape)[0]
print 'cents=', l

# full tiled meshes LxN^3
fth = np.tile(np.reshape(mh,N**3),(l,1))
fts = np.tile(np.reshape(ms,N**3),(l,1))
ftv = np.tile(np.reshape(mv,N**3),(l,1))
# tiled reduced meshes LxN^3
rth = np.tile(rh,(N**3,1)).T
rts = np.tile(rs,(N**3,1)).T
rtv = np.tile(rv,(N**3,1)).T
# distance matrix LxN^3
hdist = np.square(rth-fth)
sdist = np.square(rts-fts)
vdist = np.square(rtv-ftv)
dist = np.sqrt(hdist+sdist+vdist)
# reduced hsv cubes at min distances NxNxN
min_idx = np.argmin(dist, 0)
chue = rh[min_idx]
csat = rs[min_idx]
cval = rv[min_idx]
cubeh = chue.reshape(N,N,N)
cubes = csat.reshape(N,N,N)
cubev = cval.reshape(N,N,N)

# show cents
cimg = np.dstack((cubeh,cubes,cubev)). squeeze() . reshape(N*2,N*N/2,3)

plt.imshow(hsv_to_rgb(cimg))
plt.show()
plt.clf()

# test image
img = Image.open('orig/kueche.jpg')
img.thumbnail((256,256))
ary = np.asarray(img)/255.0
hsv = rgb_to_hsv(ary)
#
hsv = np.copy(hsv)
idx = ((hsv*N). astype(np.int)) . reshape(-1,3)

# output hsv matrix MxMx3
outh = cubeh[idx[:,0],idx[:,1],idx[:,2]]
outs = cubes[idx[:,0],idx[:,1],idx[:,2]]
outv = cubev[idx[:,0],idx[:,1],idx[:,2]]
out_hsv = np.dstack((outh.reshape(256,256), outs.reshape(256,256), outv.reshape(256,256)))
out = hsv_to_rgb(out_hsv)

plt.imshow(out)
plt.show()
plt.clf()

