import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt
from math import ceil, sqrt, pi
import pdb; __b = pdb.set_trace

# TODO:
# anzahl center vorgeben (ok)
# k nearest neighbor
# check meshgrid indexing

N = 8
#minh = 0.05 # min freq in histogram
#dmax = 0.8 # max dist from nearest neighbr
#dhmin = 0.0 # min freq of nearest neighbr
gain = 0.0
plot = True 
lense = False    

in_img = 'orig/pond.jpg'
out_img = 'orig/kueche.jpg'

N3 = N**3

# h,s,v: NxNxN
# projects to 2d plane
def plot_cube2d(h,s,v,align=None):
	if not plot:
		return 
	l = h. flatten(). size
	r = sqrt(l)
	if align is not None:
		n = ceil(r / align) * align
	else: 
		n = ceil(r)
	m = ceil(l/n)
	c = np.dstack((h.flatten(), s.flatten(), v.flatten())). squeeze()
	cube = np.copy(c)
	cube.resize(m,n,3)
	print 'cube:', cube.shape
	plt.imshow(hsv_to_rgb(cube), interpolation='none')
	plt.show()
	plt.clf()

# 
def calcCube(ary, hperc, dmax, dhmin):
	# get hue and saturation
	hsv = rgb_to_hsv(ary)
	hsv = np.reshape(hsv, (-1,3))

	# range=((0,1), (0,1), (0,1))
	hist, edges = np.histogramdd(hsv, bins=(N,N,N), normed=False )
	hist = hist / np.max(hist)
	if plot:
		plt.hist(hist.flatten(),log=False, bins=25)
		plt.show(); plt.clf()
		
	# reduce histogram
	hidx = np.argsort(hist, axis=None)
	hidx=hidx[450:]
	rhist = hist.flatten()[hidx]

	#pdb.set_trace()

	# full hsv meshes NxNxN
	ax = np.arange(0.0, 1.0, 1.0/N)
	mh,ms,mv = np.meshgrid(ax,ax,ax)

	hh,ss,vv = mh.flatten(),ms.flatten(),mv.flatten()
	mh,ms,mv = ss,hh,vv #shv

	# hsv o
	# hvs -
	# shv +
	# svh -
	# vhs -
	# vsh -

	# show pure cube
	#plot_cube2d(mh,ms,mv, 16)

	# reduces hsv meshes sized 1xL
	#perc = np.percentile(mh.flatten(),minh)
	#print 'perc=', perc
	rh = mh[hidx]
	rs = ms[hidx]
	rv = mv[hidx]
	l = (rh.shape)[0]

	# show cents
	#plot_cube2d(rh,rs,rv)

	# full tiled meshes LxN^3
	fth = np.resize(mh,(l,N**3))
	fts = np.resize(ms,(l,N**3))
	ftv = np.resize(mv,(l,N**3))
	# tiled reduced meshes LxN^3
	rth = np.resize(rh, (N**3, l)).T
	rts = np.resize(rs, (N**3, l)).T
	rtv = np.resize(rv, (N**3, l)).T
	# tiled reduced frequencies LxN^3
	#hm = hist[hist>=perc]
	#hhist = np.resize(hm, (N**3, l)).T

	# distance matrix LxN^3
	hdist = np.square(rth-fth)
	sdist = np.square(rts-fts)
	vdist = np.square(rtv-ftv)
	dist = np.sqrt((hdist+sdist+vdist)/3.0)

	#hdist = (1-dist) #* (hhist)
	#pdb.set_trace()
	min_idx = np.argsort(dist, 0)
	min_idx=min_idx[-3:]
	#pdb.set_trace()

	# weigh color with min distance
	# simple: wt=1:orig wt=0:filt
	#min_dist = np.power(np.amin(dist, 0) / sqrt(3.0), 2.0)
	#wd = np.sign(dmax-np.amax(hdist, 0))/2+0.5
	#wh = np.sign(hist.flatten()-dhmin)/2+0.5
	# (cos(x^2*pi)/2+0.5)^4
	#wh = np.power(np.cos(np.power(hist.flatten(),2.0)*pi)/2.0+0.5, 4.0)
	# membership (1=filter,0=orig)
	mu = 1 #np.amax(hdist,0)
	nu = 0
	
	cmean = np.mean(rh[min_idx], axis=0)
	smean = np.mean(rs[min_idx], axis=0)
	vmean = np.mean(rv[min_idx], axis=0)
	__b()
	chue = cmean*mu + mh*nu * gain
	csat = smean*mu + ms*nu * gain
	if not lense:
		cval = vmean * mu + mv*nu * gain
	else: 
		#cval = np.power(hist.flatten(), 0.2)
		cval = mu
	cubeh = chue.reshape(N,N,N)
	cubes = csat.reshape(N,N,N)
	cubev = cval.reshape(N,N,N)

	# show weighted filter cube
	plot_cube2d(cubeh,cubes,cubev,32)
	
	return ((cubeh, cubes, cubev), l)

def applyCube(ary, cube):
	cubeh,cubes,cubev = cube
	sh = ary.shape[0:2]
	hsv = rgb_to_hsv(ary)
	idx = ((hsv*N*0.9999). astype(np.int)) . reshape(-1,3)

	# output hsv matrix MxMx3
	outh = cubeh[idx[:,0],idx[:,1],idx[:,2]]
	outs = cubes[idx[:,0],idx[:,1],idx[:,2]]
	outv = cubev[idx[:,0],idx[:,1],idx[:,2]]
	out_hsv = np.dstack((outh.reshape(sh), outs.reshape(sh), outv.reshape(sh)))
	out = hsv_to_rgb(out_hsv)

	return out

if __name__=='__main__':
	#
	# input image
	img = Image.open(in_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0
	# minh, dmax, dhmin
	(cube, nc) = calcCube(ary, 20, 0.0, 0.0)
	print 'cents=', nc

	#
	# test image
	img = Image.open(out_img)
	img.thumbnail((256,256))
	ary = np.asarray(img)/255.0

	out = applyCube(ary, cube)

	plt.imshow(out)
	plt.show()
	plt.clf()

