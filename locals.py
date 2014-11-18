# finding local maxima
import numpy as np
import matplotlib.pyplot as plt

range = [[-0.0,1.0],[-0.5,1.5]]
range2 = [-0.0,1.0,-0.5,1.5]

def show_hist(h,xi,yi):
	plt.imshow(h, interpolation='none', cmap='hot', origin='lower', extent=range2)
	plt.show(); plt.clf()

# generate some data
def gen_data(means, Nrows=500):
	x = np.zeros( (1, len(means[1])) )
	for m in means:
		xn = np.random.normal(0.0, 0.1, (Nrows, len(means[0]))) + m
		x = np.concatenate((x,xn))
	return x[1:,...]

data = gen_data([(0.2,0.8), (0.6,0.5)])
x = data[:,0]
y = data[:,1]

plt.scatter(x,y)
plt.axis(range2)
plt.gca().set_aspect('equal')
plt.show(); plt.clf()

# calc distribution
(hist,yi,xi) = np.histogram2d(x,y, bins=15, range=range)
hist /= np.max(hist)
hist = hist.T
show_hist(hist,xi,yi)

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

min_hist = np.zeros_like(hist)
for dr in [-1,0,1]:
	for dc in [-1,0,1]:
		if dr==0 & dc == 0: continue 
		d_hist = hist[shift_indices(dr,dc)]
		min_hist[shift_indices(-dr,-dc)] += d_hist * - 0.125

hist0 = hist + min_hist
hist1 = np.maximum(hist0, 0)
show_hist(hist1, xi, yi)

loc_mins = np.zeros_like(hist)
for dr in [-1,0,1]:
	for dc in [-1,0,1]:
		if dr==0 and dc == 0: continue
		d_hist = hist[shift_indices(dr,dc)]
		new_layer = np.zeros_like(hist)
		new_layer[shift_indices(-dr,-dc)] = d_hist
		loc_mins = np.dstack((loc_mins, new_layer))
loc_mins = loc_mins[:,:,1:]

loc_hmin = np.all(hist[...,None] > loc_mins, axis=-1)
show_hist(loc_hmin, xi, yi)
idx = np.argwhere(loc_hmin)
print "idx:", idx
print xi[idx[:,0]], yi[idx[:,1]]

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
