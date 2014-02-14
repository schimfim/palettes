from random import sample

import Image
import ImageDraw
from math import sqrt, exp, cos, pi, log, acos
from time import time
from random import random, uniform
from copy import deepcopy
from logging import warning

stats = False 

def sum_rgb(a,b):
	return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

def norm(v):
	length = sqrt(sum([a**2 for a in v]))
	if length==0.0: length = 0.001
	vn = [a/length for a in v]
	return vn

def edist(a, b):
	return sqrt(sum([(x[0]-x[1])**2 for x in zip(a,b)])/len(a))

sigma = 40.0
def act(dist):
    f = exp(sigma*(1-dist))
    #f = (cos((dist**3.0)*pi)/2+0.5)**20.0
    return f
    
# softmax
def memb(color, protov, distv):	
	n = len(protov)
	a = map(act, distv)
	den = sum(a)
	f = [b / den for b in a]
	
	return f

def cluster(img, n, eps=0.001, runs=50, k=0):
	idata = img.getdata()
	if k==0:
		k = max(1000, len(idata)/10)
		warning('Set k to %i', k)
	smp = sample(idata, k)
	minc = reduce(min, smp)
	# convert samplebto float(0..1)
	smp = [norm((c[0]/255.0, c[1]/255.0, c[2]/255.0)) for c in smp]
	# init vectors
	protov = [[uniform(0.0,1.0), uniform(0.0,1.0), uniform(0.0,1.0)] for i in range(n)]
	#print protov
	#distv = [0.0]*n
	f = [0.0]*n
	last_sum = 0.0
	
	print 'clustering...'
	tic = time()

	# todo: rewrite loops pythonically
	for run in range(runs):
		new_protos = [[0.0,0.0,0.0] for i in range(n)]
		sum_f = [0.0]*n
		for c in smp:
			distv = [edist(c, p) for p in protov]
			f = memb(c, protov, distv)
			#print ['{:.2f},'.format(fc) for fc in f]
			if sum(f)-1.0 > 0.00001:
				raise Exception('f not 1', sum(f))
			for p,fc in enumerate(f):
				sum_f[p] += fc
				for i in [0,1,2]:
					new_protos[p][i] += c[i]*fc
		
		for p,fc in enumerate(sum_f):
			for i in [0,1,2]:
				new_protos[p][i] /= fc
		dproto = [abs(cp[i]-cn[i])/3/n for i in [0,1,2] for (cp,cn) in  zip(protov,new_protos)]
		protov = new_protos
		if run % 5 == 0:
			print 'run: {}'.format(run)
			print 'delta=', sum(dproto)
		if sum(dproto) < eps: break 
	
	toc = time()
	dt = toc-tic
	print '...done in {:.1f} secs'.format(dt)
	
	return protov

def load(filename, size=(256,256)):
	img = Image.open(filename)
	img.thumbnail(size, Image.ANTIALIAS)
	return img

if __name__ == '__main__':
	from palettes import draw_palette
	img = load('orig/pond.jpg')
	protov = cluster(img, 8)
	pal_img = draw_palette({'pnam':protov})
	pal_img.show()
