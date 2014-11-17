# finding local maxima
import numpy as np
import matplotlib.pyplot as plt

def show_hist(h):
	plt.imshow(h, interpolation='none', cmap='hot')
	plt.show(); plt.clf()

# generate some data
Nr = 500
x1 = np.random.normal(0.2, 0.2, Nr)
y1 = np.random.normal(0.2, 0.1, Nr)
x2 = np.random.normal(0.8, 0.1, Nr)
y2 = np.random.normal(0.5, 0.1, Nr)
x = np.hstack((x1,x2))
y = np.hstack((y1,y2))

plt.scatter(x,y)
plt.show(); plt.clf()

# calc distribution
(hist,_,_) = np.histogram2d(x,y, bins=15)
hist /= np.max(hist)
show_hist(hist)

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
show_hist(hist1)

loc_mins = np.empty_like(hist)
for dr in [-1,0,1]:
	for dc in [-1,0,1]:
		if dr==0 and dc == 0: continue
		d_hist = hist[shift_indices(dr,dc)]
		new_layer = np.zeros_like(hist)
		new_layer[shift_indices(-dr,-dc)] = d_hist
		loc_mins = np.dstack((loc_mins, new_layer))

loc_hmin = np.all(hist[...,None] > loc_mins, axis=-1)
show_hist(loc_hmin)
