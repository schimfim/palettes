# finding local maxima
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

NBINS = 15
NSAMPLES = 500 # smpls per cluster

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

def find_peaks(data, bins=NBINS):
	(h3, edges) = np.histogramdd(data3, bins=bins)
	h3 /= np.max(h3)
	edg = np.vstack(edges).T
	
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
		m = np.logical_and(data3 > e0, data3 < e1)
		ma = np.all(m, axis=1)
		mcents[row,:] = np.mean(data3[ma,:],axis=0)
		
	sizes = h3[hist1.nonzero()]
	return mcents, edg, sizes

def cutoff_cents(mcents, sizes):
	print np.vstack((mcents.T ,sizes.T)).T

	# remove small clusters
	print 'centers > 0.5:'
	ridx = sizes >= 0.5
	rcents = mcents[ridx,:]
	rsizes = sizes[ridx]
	#ridx_full = idx_full[ridx,:]

	return rcents, rsizes

###
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

mcents, edg, sizes = find_peaks(data3)
ax.scatter(mcents[:,0], mcents[:,1], mcents[:,2], c='b',s=100,alpha=0.8)

rcents, rsizes = cutoff_cents(mcents, sizes)
print np.vstack((rcents.T ,rsizes.T)).T
ax.scatter(rcents[:,0], rcents[:,1], rcents[:,2], c='r',s=400)


# todo: join large clusters using weighted means as center (to clear border cases)

#
plt.show(); plt.clf()
