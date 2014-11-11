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
S = 1
MAXN = 5

# simple colors
#cents = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
#cents = np.random.rand(5, 3)
#cents = np.array([[1.0,1.0,1.0], [0.0,0.0,0.0]])
#cents = np.array([[1.0,1.0,1.0], [0.0,0.0,0.0], [1.0,0.0,0.0]])

def calcHist(ary, top_h_perc=0.05):
	a = np.reshape(ary, (-1,3))
	N3 = N*N*N

	range = None 
	range=((0,1), (0,1), (0,1))
	hist, edges = np.histogramdd(a, bins=(N,N,N), normed=False, range=range )
	hist = hist / np.max(hist)
	if plot_all:
		plt.hist(hist.flatten(),log=True, bins=25)
		plt.show(); plt.clf()
		
	# reduce histogram
	hidx = np.argsort(hist, axis=None)
	hidx=hidx[(N3-N3*top_h_perc):]
	histi = hist.flatten()[hidx]
	__b()
	
	return (hidx, histi)
	
def calcCents(ary, top_h_perc):
	(hidx, histi) = calcHist(ary, top_h_perc)

	# full hsv meshes NxNxN
	ax = np.arange(0.0, 1.0, 1.0/N)
	mh,ms,mv = np.meshgrid(ax,ax,ax)

	hh,ss,vv = mh.flatten(),ms.flatten(),mv.flatten()
	mh,ms,mv = ss,hh,vv #shv
	# reduces hsv meshes sized 1xL
	rh = mh[hidx]
	rs = ms[hidx]
	rv = mv[hidx]
	l = (rh.shape)[0]
	cents = np.dstack((rh,rs,rv)).squeeze()
	
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
	in_img = 'orig/sunset.jpg'
	out_img = 'orig/kueche.jpg'
	
	# input image
	img = Image.open(in_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0
	(cents, nc) = calcCents(ary, 0.1)
	print 'cents=', nc
	
	# calc distribution properties
	(hidx, hist) = calcHist(ary, 0.1)
	# distance matrix
	cts = cents[..., None]
	d = np.sqrt(np.power(cts.T - cts, 2.0)/3.0)
	dist = np.sum(d, 1)

	if plot_all:
		plt.imshow(dist, interpolation='none', cmap='gray'); plt.show(); plt.clf()
	spread = np.mean(dist, 0)
	#spread /= np.max(spread)
	score = ((spread >= 0.65) & (hist > 0.0)) | (hist >= 1.0)
	ncents = cents[score]
	print 'ncents=', ncents.shape[0]

	if plot:
		plt.imshow(ncents[None,...], interpolation='none'); plt.show(); plt.clf()
	__b()
	

	# test image
	img = Image.open(out_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0

	out = applyCents(ary, ncents, 4.0)

	plt.imshow(out, interpolation='none')
	plt.show()
	plt.clf()

