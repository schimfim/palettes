import Image
import ImageDraw
from math import sqrt, exp, cos, pi, log, acos
from time import time

# module config
stats = False  
xf = 1.0
defcol = (0.0, 0.0, 0.0)

def sum_rgb(a,b):
	return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

def edist(a, b):
	return sqrt(sum([(x[0]-x[1])**2 for x in zip(a,b)])/len(a))

def fshift(color, protov, distv):	
	n = len(protov)
	f = [(cos((d**fc)*pi)/2+0.5)**slope for (d, fc) in zip(distv,focus)]
	# logistic function:
	#f = [1-1/(1+exp(-(d-1/focus)*slope)) for d in distv]

	#cols = [[color[i]*(1-fc)/n*xf+ proto[i]*fc for i in [0,1,2]] for (fc, proto) in zip(f, protov)]
	# use default color:
	cols = [[color[i]*(1-fc)/n*xf + defcol[i]*(1-fc)/n*(1-xf) + proto[i]*fc for i in [0,1,2]] for (fc, proto) in zip(f, protov)]
	avg = reduce(sum_rgb, cols)
	if stats:
		for i,v in enumerate(f):
			mmemb[i] += v

	return avg

sigma = 8.0
def act(dist):
    return exp(sigma*(1-dist))
    
# softmax
def fshift2(color, protov, distv):	
	n = len(protov)
	a = map(act, distv)
	den = sum(a)
	f = [b / den for b in a]
	
	cols = [[color[i]*(1-fc)/n*xf+proto[i]*fc for i in [0,1,2]] for (fc, proto) in zip(f, protov)]
	avg = reduce(sum_rgb, cols)

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

mdist = [] # stats
def adapt(img, pal, slp, foc):
	newi = img.copy()
	idata = newi.getdata()
	global ppal, slope, focus
	ppal = pal
	slope = slp
	focus = foc
	global mdist, mmemb
	mdist = [0.0]*len(ppal)
	mmemb = [0.0]*len(ppal)
	# TODO: use logger
	print 'processing...'
	tic = time()
	i2 = map(xform, idata)
	toc = time()
	dt = toc-tic
	print '...done in {:.3f} secs'.format(dt)
	newi.putdata(i2)
	if stats:
		mdist = [sd / len(idata) for sd in mdist]
		mmemb = [sd / len(idata) for sd in mmemb]
		print 'mdist:', ['{:.2f}'.format(md) for md in mdist]
		print 'mmean:', ['{:.2f}'.format(md) for md in mmemb]
	return newi

#
# image analysis

from random import sample
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

#
# palette calculation

def calc_palette(img, ncol=8):
	out = img.convert("P", palette = Image.ADAPTIVE, colors=ncol)
	pal = out.getpalette()
	spal = [v/255.0 for v in pal[0:ncol*3]]
	tuples = zip(*[iter(spal)]*3)
	return tuples

#
# image utilities

def load(filename, size=(256,256)):
	img = Image.open(filename)
	img.thumbnail(size, Image.ANTIALIAS)
	return img

import glob, os
def load_all(path, size=(256,256)):
	imgs = []
	for infile in glob.glob(path + "/*.jpg"):
		imgs.append(load(infile, size))
	return imgs

def save_all(imgs, path):
	for i,img in enumerate(imgs):
		fname = path + "/img_" + str(i) + ".jpg"
		img.save(fname, 'JPEG')
		
def draw_palette(pals):
	"""
	Draw palettes in pals onto one single
	Image.
	A palette is a dict with "name:[rgb...]"
	Usage: draw_palette(a,b,...)
	   or: draw_palette(*pals)
	Returns Image
	"""
	h = 50
	vals = pals.values()
	names = pals.keys()
	img = Image.new('RGB', (256,h*len(vals)))
	drw = ImageDraw.Draw(img)
	for p in range(len(vals)):
		cols = len(vals[p])
		w = 256 / cols
		for i in range(cols):
			x = i * w
			y = p * h
			pal = vals[p][i]
			col = (int(pal[0]*255.0), int(pal[1]*255.0), int(pal[2]*255.0))
			drw.rectangle([(x,y),(x+w,y+h)], fill=col)
			drw.text((0,y), names[p])
	img.show()
	return img

