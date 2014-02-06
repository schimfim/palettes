from random import sample

import Image
import ImageDraw
from math import sqrt, exp, cos, pi, log, acos
from time import time

stats = False 

def sum_rgb(a,b):
	return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

def edist(a, b):
	return sqrt(sum([(x[0]-x[1])**2 for x in zip(a,b)])/len(a))

def memb(d,fc,slope):
	# membership function
	f = (cos((d**fc)*pi)/2+0.5)**slope

	return f

def fshift(color, protov, distv):	
	n = len(protov)
	f = [memb(d,fc,slope) for (d, fc) in zip(distv,focus)]

	cols = [[color[i]*(1-fc)/n*xf + defcol[i]*(1-fc)/n*(1-xf) + proto[i]*fc for i in [0,1,2]] for (fc, proto) in zip(f, protov)]
	avg = reduce(sum_rgb, cols)
	if stats:
		for i,v in enumerate(f):
			mmemb[i] += v

	return avg

def xform(c):
	# convert to float(0..1)
	v = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
	
	# calc memberships
	dist = [0.0]*len(ppal)
	for i, prot in enumerate(ppal):
		dist[i] = edist(v, prot)
		if not stats: continue
		mdist[i] += dist[i]
	
	# shift color
	s = fshift(v, ppal, dist)
	
	# convert to int(0..255)
	v = (int(s[0]*255.0), int(s[1]*255.0), int(s[2]*255.0))
	return v


def analyse(img, pal, gain=0.5, k=1000, nperc=0.1):
	idata = img.getdata()
	smp = sample(idata, k)
	print 'analysing...'
	tic = time()

	d = []
	empty = []
	d = [empty[:] for i in pal]
	for c in smp:
		# convert to float(0..1)
		v = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
		for i,p in enumerate(pal):
			# calc distance
			dist = edist(v, p)
			d[i].append(dist)
	d = [sorted(di) for di in d]
	toc = time()
	dt = toc-tic
	print '...done in {:.3f} secs'.format(dt)
	perc = [di[int(k*nperc)] for di in d]
	focus = [(log(acos(2.0*gain**(1/20.0)-1)) - log(pi))/log(p) for p in perc]
	print 'nthperc={:.2f}, gain={:.2f}'.format(nperc, gain)
	print 'perc:', ['{:.3f}'.format(p) for p in perc]
	print 'focus:', ['{:.3f}'.format(f) for f in focus]
	
	return focus

