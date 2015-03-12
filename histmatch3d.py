'''
Non parametric 3d-histogram matching
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import Image

NBINS = 16
BASE = 0.01

bin1 = np.linspace(0,1,NBINS+1)
bins = (bin1, bin1, bin1)

# utility
def getax():
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	return ax
	
def im3(d):
	plt.clf()
	for i in (0,1,2):
		plt.subplot(1,3,i+1)
		plt.imshow(np.sum(d,axis=i), cmap='hot', extent=[0,1,1,0], interpolation='none')
	plt.show()

def im2(d,n=int(NBINS/2)):
	plt.clf()
	plt.subplot(131)
	plt.imshow(d[n,:,:], cmap='hot', extent=[0,1,1,0], interpolation='none')
	plt.subplot(132)
	plt.imshow(d[:,n,:], cmap='hot', extent=[0,1,1,0], interpolation='none')
	plt.subplot(133)
	plt.imshow(d[:,:,n], cmap='hot', extent=[0,1,1,0], interpolation='none')
	plt.show()

'''
Reference signal
'''
N = 128
#means = [[0.1, 0.6, 0.2], [0.2, 0.0, 0.9], [0.7, 0.6, 0.6], [0.0,0.3,0.1]]
#means = [[0.2, 0.4, 0.6]]
means = [[0.9, 0.1, 0.2], [0.0, 0.8, 0.5]]
cov1 = np.diag([1]*3) * 0.05
M = len(means)

ref_s = np.random.multivariate_normal(means[0], cov1, N**2/M)
for m in means[1:]:
	s = np.random.multivariate_normal(m, cov1, N**2/M)
	ref_s = np.concatenate((ref_s, s), axis=0)

ref_s = ref_s.reshape((N, N, 3))

plt.clf()
plt.title('Reference signal')
plt.imshow(ref_s,extent=[0,1,1,0], interpolation='none')
plt.show()

'''
Reference distribution
'''
ref_n, foo = np.histogramdd(ref_s.reshape((-1,3)), bins=bins)
ref_n = ref_n / float(ref_s.shape[0]) + BASE
x = np.linspace(0,1,NBINS)
X,Y = np.meshgrid(x,x, indexing='ij')
# plot
im3(ref_n)

'''
Integrate and invert reference distro
in three dimensions
'''
sn = np.sum(ref_n, axis=0, keepdims=True)
sn[sn == 0] = 1
ref_intx = np.cumsum(ref_n, axis=0)
#im2(ref_intx)
ref_intx /= sn
im2(ref_intx)
sn = np.sum(ref_n, axis=1, keepdims=True)
sn[sn == 0] = 1
ref_inty = np.cumsum(ref_n, axis=1)
#im2(ref_inty)
ref_inty /= sn
im2(ref_inty)
sn = np.sum(ref_n, axis=2, keepdims=True)
sn[sn == 0] = 1
ref_intz = np.cumsum(ref_n, axis=2)
#im2(ref_intz)
ref_intz /= sn
im2(ref_intz)
# Invert
x = np.linspace(0,1,NBINS)
y = np.linspace(0,1,NBINS)
z = np.linspace(0,1,NBINS)
invx = np.zeros((NBINS,NBINS,NBINS))
invy = np.zeros((NBINS,NBINS,NBINS))
invz = np.zeros((NBINS,NBINS,NBINS))
for k,zk in enumerate(z):
	for j,yj in enumerate(y):
		for i,xi in enumerate(x):
			l = np.argwhere(xi > ref_intx[:,j,k])
			if len(l) > 0: invx[i,j,k] = x[l[-1]]
			l = np.argwhere(yj > ref_inty[i,:,k])
			if len(l) > 0: invy[i,j,k] = y[l[-1]]
			l = np.argwhere(zk > ref_intz[i,j,:])
			if len(l) > 0: invz[i,j,k] = z[l[-1]]

im3(invx)
im3(invy)
im3(invz)
#NEXT: invxyz betrachten, ist immer einer falsch...

'''
Input signal
'''
# calc
x = np.linspace(0,1,N)
X,Y = np.meshgrid(x,x, indexing='ij')
hsv = np.dstack((X,Y,np.ones_like(X)*0.8))
in_s = hsv_to_rgb(hsv)
# plot
'''
plt.clf()
plt.title('Input signal')
plt.imshow(in_s,extent=[0,1,1,0], interpolation='none')
plt.show()
'''
'''
Process input
'''
# Process
idx=np.asarray(in_s*(NBINS),dtype=np.integer)
out_s0 = invx[idx[...,0], idx[...,1], idx[...,2]]
out_s1 = invy[idx[...,0], idx[...,1], idx[...,2]]
out_s2 = invz[idx[...,0], idx[...,1], idx[...,2]]
out_s = np.dstack((out_s0, out_s1, out_s2))

plt.clf()
plt.title('Output signal')
plt.imshow(out_s,extent=[0,1,1,0], interpolation='none')
plt.show()

'''
Image
'''
out_img = 'orig/kueche.jpg'
oimg = Image.open(out_img)
oimg.thumbnail((256,256))
oary = np.asarray(oimg)/255.0
# Process
idx=np.asarray(oary*(NBINS),dtype=np.integer)
out_s0 = invx[idx[...,0], idx[...,1], idx[...,2]]
out_s1 = invy[idx[...,0], idx[...,1], idx[...,2]]
out_s2 = invz[idx[...,0], idx[...,1], idx[...,2]]
out_s = np.dstack((out_s0, out_s1, out_s2))

plt.clf()
plt.title('Output image')
plt.imshow(out_s,extent=[0,1,1,0], interpolation='none')
plt.show()
