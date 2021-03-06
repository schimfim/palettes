import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt
from math import ceil, sqrt
import pdb

plot = True
plot_all = False
debug = True 
debug_all = False 

def __b(set=None):
	if (debug & (set==1)) | debug_all:
		pdb.set_trace()

# hist size
N = 8
MAXN = 5

def plot_cube2d(h,s,v,align=None):
	if not plot:
		return 
	l = h. flatten(). size
	r = sqrt(l)
	if align is not None:
		n = ceil(r / align) * align
	else: 
		n = ceil(r)
	m = ceil(l/n)
	c = np.dstack((h.flatten(), s.flatten(), v.flatten())). squeeze()
	cube = np.copy(c)
	cube.resize(m,n,3)
	plt.imshow(cube, interpolation='none')
	plt.show()
	plt.clf()

# returns index tuples for shifting
# source and destination matrices
def shift_indices(ds, dr, dc):
	row_from = None 
	row_to = None 
	if dr==1: row_from = 1
	if dr==-1: row_to = -1
	col_from = None 
	col_to = None 
	if dc==1: col_from = 1
	if dc==-1: col_to = -1
	slc_from = None
	slc_to = None
	if ds==1: slc_from = 1
	if ds==-1: slc_to = -1
	src_idx = np.s_[slc_from:slc_to, row_from:row_to, col_from:col_to]
	
	return src_idx
	
def min_hist3d(h):
	min_hist = np.zeros_like(h)
	for ds in [-1,0,1]:
		for dr in [-1,0,1]:
			for dc in [-1,0,1]:
				if dr==0 & dc == 0 & ds==0: continue
				d_hist = h[shift_indices(ds,dr,dc)]
				min_hist[shift_indices(-ds,-dr,-dc)] += d_hist * - 0.25

	hist0 = h + min_hist
	hist1 = np.maximum(hist0, 0)
	return hist1

def min_hist3d_2(h):
	min_hist = np.zeros_like(h[...,None ])
	for (ds,dr,dc) in zip([-1,1,0,0,0,0], [0,0,-1,1,0,0], [0,0,0,0,-1,1]):
		d_hist = h[shift_indices(ds,dr,dc)]
		new_layer = np.zeros_like(h)
		new_layer[shift_indices(-ds,-dr,-dc)] = d_hist
		__b(0)
		min_hist = np.concatenate((min_hist, new_layer[...,None]), axis=3)
		#min_hist = np.dstack((min_hist, new_layer))

	print 'min_hist.shape:', min_hist.shape
	hist1 = np.all(h[...,None] > min_hist, axis=3)
	print 'Found local peaks:', np.count_nonzero(hist1)
	__b(0)
	#h[hist1] = 1.0
	h[hist1 == False] = 0.0
	return h

def calcHist(ary, nhist=50):
	a = np.reshape(ary, (-1,3))
	N3 = N*N*N

	range = None 
	range=((0,1), (0,1), (0,1))
	hist, edges = np.histogramdd(a, bins=(N,N,N), normed=False, range=range )
	hist = hist / np.max(hist)
	hist = min_hist3d_2(hist)
	if plot_all:
		plt.hist(hist.flatten(),log=True, bins=25)
		plt.show(); plt.clf()
		
	# reduce histogram
	nz = hist.flatten() != 0
	hidx = np.argsort(hist, axis=None)
	hidx = hidx[-nhist:]
	__b()
	hidx = np.setdiff1d(hidx, nz)
	hidx = hidx[nz[hidx]]
	histi = hist.flatten()[hidx]
	__b()
	
	return (hidx, histi)
	
def reduc(cents, hist):
	# distance matrix
	cts = cents[..., None]
	d = np.sqrt(np.power(cts.T - cts, 2.0)/3.0)
	dist = np.sum(d, 1)

	if plot_all:
		plt.imshow(dist, interpolation='none', cmap='gray'); plt.show(); plt.clf()
	# 
	# select centers
	# falsch: score = (dist.T * hist).T
	score = dist #* hist
	idx = np.array([np.argmax(hist)])
	rng = np.arange(hist.shape[0])
	while idx.size < MAXN:
		cols = np.setdiff1d(rng, idx)
		dcent = score[idx][:,cols]
		dsum = np.sum(dcent, 0)
		dmax = np.argmax(dsum)
		idx = np.append(idx, dmax)
	ncents = cents[idx]
	
	print 'ncents=', ncents.shape[0]

	if plot:
		plt.imshow(ncents[None,...], interpolation='none'); plt.show(); plt.clf()
	__b()
	return ncents
	
def calcCents(ary, nhist=50):
	(hidx, histi) = calcHist(ary, nhist)

	# full hsv meshes NxNxN
	ax = np.arange(0.0, 1.0, 1.0/N)
	mh,ms,mv = np.meshgrid(ax,ax,ax)

	hh,ss,vv = mh.flatten(),ms.flatten(),mv.flatten()
	mh,ms,mv = ss,hh,vv #shv
	# reduces hsv meshes sized 1xL
	rh = mh[hidx]
	rs = ms[hidx]
	rv = mv[hidx]
	cents = np.dstack((rh,rs,rv)).squeeze()
	
	cents = reduc(cents, histi)
	l = (cents.shape)[0]
	
	__b()
	if plot:
		plt.imshow(cents[None, ...], interpolation='none'); plt.show(); plt.clf()
	
	return (cents, l)


def applyCents(ary, cents, CT=4.0):
	__b()
	nc = len(cents)
	sh = ary.shape
	N = sh[0] * sh[1]
	print "sh:", sh
	a = np.reshape(ary, (-1,3))
	dist = np.zeros((N,nc))
	for (i,c) in enumerate(cents):
		dc = np.power(a - c, 2.0)
		dist[:,i] = np.sqrt(np.sum(dc,1)/nc)
	sdist = np.sum((1/dist)**CT, 1)
	sdist = np.resize(sdist, (nc,N)).T
	print "sdist:", sdist.shape
	mu = (1/dist)**CT / sdist
	print "mu:", mu.shape
	out = np.dot(mu, cents)
	return out.reshape(sh)

if __name__=='__main__':
	in_img = 'orig/rgb.jpg'
	out_img = 'orig/sylt.jpg'
	
	# input image
	img = Image.open(in_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0
	(cents, nc) = calcCents(ary)
	print 'cents=', nc
	
	# test image
	img = Image.open(out_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0

	out = applyCents(ary, cents, 3.0)

	plt.imshow(out, interpolation='none')
	plt.show()
	plt.clf()

