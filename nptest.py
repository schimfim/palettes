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

# simple colors
#cents = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
#cents = np.random.rand(5, 3)
cents = np.array([[1.0,1.0,1.0], [0.0,0.0,0.0]])
#cents = np.array([[1.0,1.0,1.0], [0.0,0.0,0.0], [1.0,0.0,0.0]])
NC = len(cents)

def applyCents(ary, cents):
	__b()
	sh = ary.shape
	N = sh[0] * sh[1]
	print "sh:", sh
	a = np.reshape(ary, (-1,3))
	dist = np.zeros((N,NC))
	for (i,c) in enumerate(cents):
		dc = np.power(a - c, 2.0)
		dist[:,i] = np.sqrt(np.sum(dc,1)/NC)
	sdist = np.sum(1/dist, 1)
	sdist = np.resize(sdist, (NC,N)).T
	print "sdist:", sdist.shape
	mu = (1/dist) / sdist
	print "mu:", mu.shape
	out = np.dot(mu, cents)
	return out.reshape(sh)

if __name__=='__main__':
	in_img = 'orig/pond.jpg'
	out_img = 'orig/kueche.jpg'

	#
	# test image
	img = Image.open(out_img)
	img.thumbnail((512,512))
	ary = np.asarray(img)/255.0

	out = applyCents(ary, cents)

	plt.imshow(out)
	plt.show()
	plt.clf()
