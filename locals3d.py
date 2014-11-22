# finding local maxima
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

NBINS = 20
NSAMPLES = 500 # smpls per cluster
	
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

(h3, edges) = np.histogramdd(data3, bins=NBINS)
h3 /= np.max(h3)

min_hist = np.zeros_like(h3)[None,...]
#for (ds,dr,dc) in zip([-1,1,0,0,0,0], [0,0,-1,1,0,0], [0,0,0,0,-1,1]):
for dr in [-1,0,1]:
	for dc in [-1,0,1]:
		for ds in [-1,0,1]:
			if dr==0 and dc == 0 and ds==0: continue 
			d_hist = h3[shift_indices3d(ds,dr,dc)]
			new_layer = np.zeros_like(h3)
			new_layer[shift_indices3d(-ds,-dr,-dc)] = d_hist
			#__b(0)
			min_hist = np.concatenate((min_hist, new_layer[None,...]))
			#min_hist = np.dstack((min_hist, new_layer))

print 'min_hist.shape:', min_hist.shape
hist1 = np.all(h3[None,...] > min_hist, axis=0)
print 'Found local peaks:', np.count_nonzero(hist1)

# calc centers
h3[hist1 == False] = 0.0
idx_full = np.vstack(hist1.nonzero()).T
xx = edges[0][idx_full[:,0]]
yy = edges[1][idx_full[:,1]]
zz = edges[2][idx_full[:,2]]

ax.scatter(xx,yy,zz,c='b',s=100,alpha=0.8)

np.set_printoptions(precision=3, suppress=True)
print 'centers:'
cents = np.vstack((xx,yy,zz)).T
sizes = h3[hist1.nonzero()]
print np.vstack((cents.T ,sizes.T)).T

# remove small clusters
print 'centers > 0.5:'
ridx = sizes >= 0.5
rcents = cents[ridx,:]
rsizes = sizes[ridx]
ridx_full = idx_full[ridx,:]
print np.vstack((rcents.T ,rsizes.T)).T
ax.scatter(rcents[:,0], rcents[:,1], rcents[:,2], c='r',s=400)

# todo: use average of peak histogram cell as center
for i in ridx_full:
    data3[:,0] > edges[0][i[0]] and data3[:,0] <= edges[0][i[0]+1]

# todo: join large clusters using weighted means as center (to clear border cases)

#
plt.show(); plt.clf()
