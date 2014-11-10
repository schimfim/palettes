import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt
from math import ceil, sqrt
import pdb

def nthg():
	pass

plot = True
debug = False 

if debug: __b = pdb.set_trace
else: __b = nthg

# hist size
N = 8
S = 1
CT = 4.0
MAXN = 5

# simple colors
#cents = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
#cents = np.random.rand(5, 3)
#cents = np.array([[1.0,1.0,1.0], [0.0,0.0,0.0]])
#cents = np.array([[1.0,1.0,1.0], [0.0,0.0,0.0], [1.0,0.0,0.0]])

def calcCents(ary, top_h_perc):
	a = np.reshape(ary, (-1,3))
	N3 = N*N*N

	# range=((0,1), (0,1), (0,1))
	hist, edges = np.histogramdd(a, bins=(N,N,N), normed=False, range=((0,1), (0,1), (0,1)) )
	hist = hist / np.max(hist)
	if plot:
		plt.hist(hist.flatten(),log=True, bins=25)
		plt.show(); plt.clf()
		
	# reduce histogram
	hidx = np.argsort(hist, axis=None)
	hidx=hidx[(N3-N3*top_h_perc):]

	# full hsv meshes NxNxN
	ax = np.arange(0.0, 1.0, 1.0/N)
	mh,ms,mv = np.meshgrid(ax,ax,ax)

	hh,ss,vv = mh.flatten(),ms.flatten(),mv.flatten()
	mh,ms,mv = ss,hh,vv #shv
	# reduces hsv meshes sized 1xL
	rh = mh[hidx]
	rs = ms[hidx]
	rv = mv[hidx]
	
	# prune down to MAXN cents
	while len(hidx) > MAXN:
		
	
	l = (rh.shape)[0]
	cents = np.dstack((rh.flatten(), rs.flatten(), rv.flatten())). squeeze()
	
	if plot:
		disp = np.expand_dims(cents,0)
		__b()
		plt.imshow(disp, interpolation='none'); plt.show(); plt.clf()
	__b()
	
	return (cents, l)


def applyCents(ary, cents):
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
	in_img = 'orig/pond.jpg'
	out_img = 'orig/dunes.jpg'
	
	# input image
	img = Image.open(in_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0
	(cents, nc) = calcCents(ary, 0.02)
	print 'cents=', nc
	
	# test image
	img = Image.open(out_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0

	out = applyCents(ary, cents)

	plt.imshow(out)
	plt.show()
	plt.clf()
