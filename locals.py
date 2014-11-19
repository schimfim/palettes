# finding local maxima
import numpy as np
import matplotlib.pyplot as plt

range = [[-0.0,1.0],[-0.5,1.5]]
range2 = [-0.0,1.0,-0.5,1.5]

def show_hist(h,xi,yi):
	plt.imshow(h, interpolation='none', cmap='hot', origin='lower', extent=range2)
	plt.show(); plt.clf()


# generate some samples
def gen_data(means, Nrows=500):
	Ndim = len(means[1])
	Np = len(means[0])
	print 'Gen %d points of dim %d' % (Np, Ndim)
	x = np.zeros((1, Ndim))
	for m in means:
		xn = np.random.normal(0.0, 0.1, (Nrows, Ndim)) + m
		x = np.concatenate((x,xn))
	return x[1:,...]

data = gen_data([[0.2,0.8], [0.6,0.5], [0.9,0.5]])
x = data[:,0]
y = data[:,1]

plt.scatter(x,y)
plt.axis(range2)
plt.gca().set_aspect('equal')
plt.show(); plt.clf()

# calc distribution
(hh,xi,yi) = np.histogram2d(x,y, bins=15, range=range)
hh /= np.max(hh)
hh = hh.T
show_hist(hh,xi,yi)

#
# find local maximae

# returns index tuples for shifting
# source and destination matrices
def shift_indices(dr, dc):
	row_from = None 
	row_to = None 
	if dr==1: row_from = 1
	if dr==-1: row_to = -1
	col_from = None 
	col_to = None 
	if dc==1: col_from = 1
	if dc==-1: col_to = -1
	src_idx = np.s_[row_from:row_to, col_from:col_to]
	
	return src_idx

min_hist = np.zeros_like(hh)
for dr in [-1,0,1]:
	for dc in [-1,0,1]:
		if dr==0 & dc == 0: continue 
		d_hist = hh[shift_indices(dr,dc)]
		min_hist[shift_indices(-dr,-dc)] += d_hist * - 0.125

hist0 = hh + min_hist
hist1 = np.maximum(hist0, 0)
show_hist(hist1, xi, yi)

loc_mins = np.zeros_like(hh)
for dr in [-1,0,1]:
	for dc in [-1,0,1]:
		if dr==0 and dc == 0: continue
		d_hist = hh[shift_indices(dr,dc)]
		new_layer = np.zeros_like(hh)
		new_layer[shift_indices(-dr,-dc)] = d_hist
		loc_mins = np.dstack((loc_mins, new_layer))
loc_mins = loc_mins[:,:,1:]

loc_hmin = np.all(hh[...,None] > loc_mins, axis=-1)
show_hist(loc_hmin, xi, yi)
idx = np.argwhere(loc_hmin)
print "idx:", idx
print xi[idx[:,1]], yi[idx[:,0]]

print '3D DATA'
data3 = gen_data([(0.2,0.8,0.3),
                  (0.6,0.5,0.9),
                  (0.1,0.1,0.2)])
x = data[:,0]
y = data[:,1]
z = data[:,2]

'''
min_hist = np.empty_like(h[...,None ])
for (ds,dr,dc) in zip([-1,1,0,0,0,0], [0,0,-1,1,0,0], [0,0,0,0,-1,1]):
	d_hist = h[shift_indices(ds,dr,dc)]
	new_layer = np.zeros_like(h)
	new_layer[shift_indices(-ds,-dr,-dc)] = d_hist
	__b(0)
	min_hist = np.concatenate((min_hist, new_layer), axis=3)
	#min_hist = np.dstack((min_hist, new_layer))

print 'min_hist.shape:', min_hist.shape
hist1 = np.all(h[...,None] > min_hist, axis=3)
print 'Found local peaks:', np.count_nonzero(hist1)
__b(0)
h[hist1] = 1.0
h[hist1 == False] = 0.0
'''
