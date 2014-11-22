# finding local maxima
import numpy as np
import matplotlib.pyplot as plt

range1 = [[-0.0,1.0],[-0.5,1.5]]
range2 = [-0.0,1.0,-0.5,1.5]

def show_hist(h,xi,yi, hold=False):
	plt.imshow(h, interpolation='none', cmap='hot', origin='lower', extent=range2)
	if not hold:
		plt.show(); plt.clf()
	
def gen_data(means, Nrows=500):
	Ndim = len(means[0])
	Np = len(means)
	print 'Gen %d points of dim %d' % (Np, Ndim)
	x = np.zeros((1, Ndim))
	for m in means:
		xn = np.random.normal(0.0, 0.1, (Nrows, Ndim)) + m
		x = np.concatenate((x,xn))
	return x[1:,...]

means = [[0.2,0.8], [0.6,0.5], [0.9,0.5]]
data = gen_data(means)
x = data[:,0]
y = data[:,1]

plt.scatter(x,y,s=1)
plt.axis(range2)
#plt.gca().set_aspect('equal')
#plt.show(); plt.clf()
# plot means
pmeans = np.array(means)
plt.scatter(pmeans[:,0],pmeans[:,1],c='g',s=200, marker='v')

# calc distribution
(hh,xi,yi) = np.histogram2d(x,y, bins=15, range=range1)
hh /= np.max(hh)
#hh = hh.T
# hh has x in rows!
#show_hist(hh.T,xi,yi,hold=True)

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
#show_hist(hist1.T, xi, yi)

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
#show_hist(loc_hmin.T, xi, yi)
idx = np.argwhere(loc_hmin)
print "idx:", idx
xn = (xi[idx[:,0]] + xi[idx[:,0]+1])/2
yn =  (yi[idx[:,1]] + yi[idx[:,1]+1])/2
xy = np.vstack((xn,yn)).T
plt.scatter(xn,yn,c='r',s=hh[idx[:,0],idx[:,1]]*1000, marker='o', alpha=0.5)
plt.show()
plt.clf()
