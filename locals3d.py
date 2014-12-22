# finding local maxima
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import Image

import pdb
debug = True
debug_all = False
def __b(set=None):
	if (debug & (set==1)) | debug_all:
		pdb.set_trace()

verbose = True 

NBINS = 16
NSAMPLES = 500 # smpls per cluster
MAXC = 6

np.set_printoptions(precision=3, suppress=True)

def gen_data(means, Nrows=500):
    Ndim = len(means[0])
    Np = len(means)
    print 'Gen %d points of dim %d' % (Np, Ndim)
    x = np.zeros((1, Ndim))
    for m in means:
        xn = np.random.normal(0.0, 0.1, (Nrows, Ndim)) + m
        x = np.concatenate((x,xn))
    return x[1:,...]

def shift_indices3d(ds, dr, dc):
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

def cutoff_cents(mcents, sizes, cutoff=0.01):
	#print np.vstack((mcents.T ,sizes.T)).T

	# remove small clusters
	print 'centers >', cutoff
	ridx = sizes >= cutoff
	rcents = mcents[ridx,:]
	rsizes = sizes[ridx]
	#ridx_full = idx_full[ridx,:]

	return rcents, rsizes
	
def join_cents(cents, sizes, maxc=MAXC):
	ncents = np.ma.array(cents)
	nsizes = np.ma.array(sizes)
	cts = cents[..., None]
	d = np.power(cts.T - cts, 2.0)
	dist = np.sqrt(np.sum(d, 1) / 3.0)
	distm = np.ma.masked_less_equal(dist, 0.0)
	
	n = cents.shape[0]-maxc
	print 'reduce by:', n
	if n <= 0: return cents, sizes
	for i in range(cents.shape[0]-maxc):
		idx = np.unravel_index(np.argmin(distm), distm.shape)
		idx = np.array(idx)
		__b()
		new_cent = np.mean(cents[idx],0)
		ncents[idx[0]] = new_cent
		ncents[idx[1]] = np.ma.masked
		nsizes[idx[0]] = np.sum(sizes[idx],0) 
		nsizes[idx[1]] = np.ma.masked
		#print 'new cent:', new_cent
		#print 'from items:', idx
		# mask resp. cols/rows in distance
		# matrix
		distm[idx[1],:] = np.ma.masked
		distm[:,idx[1]] = np.ma.masked
		# todo: update dist matrix...
	
	nc = ncents[np.all(ncents.mask==False, 1)]
	ns = nsizes[np.all(ncents.mask==False, 1)]
	
	return nc, ns

def find_peaks(data, bins=NBINS, cutoff=0.01):
	data = np.reshape(data, (-1,3))
	rng=((0,1),(0,1),(0,1))
	rng = None 
	#rng=None 
	(h3, edges) = np.histogramdd(data, bins=bins, range=rng)
	h3 /= np.max(h3)
	edg = np.vstack(edges).T
	
	# find local peaks
	min_hist = np.zeros_like(h3)[None,...]
	for dr in [-1,0,1]:
		for dc in [-1,0,1]:
			for ds in [-1,0,1]:
				if dr==0 and dc == 0 and ds==0: continue 
				d_hist = h3[shift_indices3d(ds,dr,dc)]
				new_layer = np.zeros_like(h3)
				new_layer[shift_indices3d(-ds,-dr,-dc)] = d_hist
				min_hist = np.concatenate((min_hist, new_layer[None,...]))
	
	print 'min_hist.shape:', min_hist.shape
	hist1 = np.all(h3[None,...] > min_hist, axis=0)
	print 'Found local peaks:', np.count_nonzero(hist1)

	# calc centers
	h3[hist1 == False] = 0.0
	idx_full = np.vstack(hist1.nonzero()).T
	cents = edg[idx_full, [0,1,2]]

	# use average of peak histogram cell as center
	mcents = np.zeros_like(cents)
	for (row,i) in enumerate(idx_full):
		e0 = edg[i, [0,1,2]]
		e1 = edg[i+1, [0,1,2]]
		#print 'edges:', e0, e1
		__b()
		m = np.logical_and(data > e0, data < e1)
		ma = np.all(m, axis=1)
		mcents[row,:] = np.mean(data[ma,:],axis=0)
		#print 'row:', mcents[row,:]
	__b()
	
	sizes = h3[hist1.nonzero()]
	
	# cutoff small clusters
	rcents, rsizes = cutoff_cents(mcents, sizes, cutoff=cutoff)
	
	# join close clusters
	jcents, jsizes = join_cents(rcents, rsizes)
	
	return jcents, jsizes

##
# src entspricht filter image
def match_cents(src, src_sizes, out, out_sizes, maxc=MAXC):
	csrc = src[..., None]
	cout = out[..., None]
	__b()
	d = np.power(cout.T - csrc, 2.0)
	dist = np.sum(d, 1)
	#volume = np.outer(src_sizes, out_sizes)
	volume = np.outer(out_sizes, src_sizes)
	distm = np.ma.array(dist / np.sqrt(volume))
	
	src_idx = []
	out_idx = []
	for i in range(src.shape[0]):
		idx = np.unravel_index(np.argmin(distm), distm.shape)
		src_idx.append(idx[0])
		out_idx.append(idx[1])
		# mask resp. cols/rows in distance
		# matrix
		distm[idx[0],:] = np.ma.masked
		distm[:,idx[1]] = np.ma.masked
	
	nsrc = src[src_idx]
	nout = out[out_idx]
	
	return nsrc, nout


def applyCents(ary, cents, CT=3.0):
	__b()
	nc = cents.shape[0]
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

# match against cents
# apply colors from outcents
def fastApplyCents(ary, cents, mu0=0.0, CT=3.0, outcents=None ):
	__b()
	if outcents==None :
		outcents=cents
	nc = cents.shape[0]
	sh = ary.shape
	N = sh[0] * sh[1]
	print "sh:", sh
	a = np.reshape(ary, (-1,3))
	dist = np.zeros((N,nc))
	for (i,c) in enumerate(cents):
		dc = np.absolute(a - c)
		dist[:,i] = np.sum(dc,1)
		# dist[:,i] = np.sqrt(np.sum(dc,1)/nc)
	sdist0 = np.sum((1/dist)**CT, 1)
	# add distance to orig
	__b()
	mu0 = np.ones(N)*mu0
	#d0 = 0.999
	#id0 = np.ones(N)*(1/d0)**CT
	id0 = - mu0 * sdist0 / (mu0 - 1)
	sdist = sdist0 + id0
	#mu0 = id0 / sdist
	#   id0 = - mu0 * sdist0 / (mu0 - 1)
	sdist = np.resize(sdist, (nc,N)).T
	print "sdist:", sdist.shape
	mu = (1/dist)**CT / sdist
	print "mu:", mu.shape
	#
	out = np.dot(mu, outcents)
	out = out + mu0[:,None] * a
	return out.reshape(sh)

###
if __name__=='__main__':
	
	'''
	print '3D DATA'
	means = [(0.2,0.8,0.3),
         (0.6,0.5,0.9),
         (0.1,0.1,0.2)]
	data3 = gen_data(means, Nrows=NSAMPLES)
	x = data3[:,0]
	y = data3[:,1]
	z = data3[:,2]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x,y,z, s=1, alpha=0.3)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	pmeans = np.array(means)
	ax.scatter(pmeans[:,0],pmeans[:,1], pmeans[:,2], c='g',s=800, marker='v')

	rcents = find_peaks(data3)
	#print np.vstack((rcents.T ,rsizes.T)).T
	ax.scatter(rcents[:,0], rcents[:,1], rcents[:,2], c='r',s=400)

	plt.show(); plt.clf()
	'''
	
	# Image data
	in_img = 'orig/pond.jpg'
	out_img = 'orig/kueche.jpg'
	#
	# input image
	img = Image.open(in_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0

	ccents, csizes = find_peaks(ary)
	plt.imshow(ccents[None, ...], interpolation='none' )
	plt.show(); plt.clf()

	# output image
	oimg = Image.open(out_img)
	oimg.thumbnail((256,256))
	oary = np.asarray(oimg)/255.0
	#out = applyCents(oary, ccents)
	outcents, outsizes = find_peaks(oary)
	nsrc, nout = match_cents(ccents, csizes, outcents, outsizes)
	
	plt.imshow(np.dstack((nsrc.T,nout.T)).T, interpolation='none' )
	plt.show(); plt.clf()
	
	out = fastApplyCents(oary, nsrc, mu0=0.2, CT=3.0)
	#out = fastApplyCents(oary, nout, outcents=nsrc, mu0=0.0, CT=3.0)
	
	plt.imshow(out, interpolation='none')
	plt.show(); plt.clf()
