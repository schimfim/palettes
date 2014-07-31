# todo:
# x-werte bei interp_c verwenden

import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt

# params
nbins = 64
size = (512,512)

def open_hsv(fname, size=(256,256)):
	img = Image.open(fname)
	img.thumbnail(size)
	ary = np.asarray(img)/255.0
	# xform to hsv
	hsv = rgb_to_hsv(ary)
	#hsv = ary
	return hsv

def cdf_c(vals, n=nbins):
	bins = np.zeros(n)
	s = 0
	for i in range(len(vals)):
		idx = int(vals[i] * (n-1))
		bins[idx] += 1
		s += 1
	for i in range(1,n):
		bins[i] = bins[i-1] + bins[i]
	for i in range(n):
		bins[i] = float(bins[i]/s)
	return bins

def interp_c(x, xt, yt):
	left = 0.0
	ly = 0.0
	it = 0
	right = xt[it]
	ry = yt[it]
	nx = len(x)
	ix = 0
	y = np.zeros(nx)
	while ix < nx:
		if x[ix] > right:
			left = xt[it]
			ly = yt[it]
			it += 1
			right = xt[it]
			ry = yt[it]
		y[ix] = (ry+ly)/2
		ix += 1
	return y

def gen_map(vals, n=nbins):
	# process hues
	yh, xh, patches = plt.hist(vals.flatten(), bins=n, range=(0,1), normed=True, cumulative=True, histtype='step')
	xhi = np.linspace(0,1,256)
	yhi = np.interp(xhi, yh, xh[1:])
	
	xch = np.linspace(0,1,n)
	xc = cdf_c(vals.flatten(), n=n)
	plt.plot(xch, xc)
	yc = interp_c(xhi, xc, xch)
	plt.plot(xhi, yhi, xhi, yc)
	return yhi

hsv = open_hsv('orig/pond.jpg')
maps = gen_map(hsv[...,0])
plt.show()

