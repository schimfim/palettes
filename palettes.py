import Image
import ImageDraw
from math import sqrt, exp, cos, pi, log
from time import time

# module config
stats = False  
xf = 1.0

def sum_rgb(a,b):
	return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

def edist(a, b):
	return sqrt(sum([(x[0]-x[1])**2 for x in zip(a,b)])/len(a))

def fshift(color, protov, distv):	
	n = len(protov)
	f = [(cos((d**focus)*pi)/2+0.5)**slope for d in distv]
	# alt.: 1-1/(1+exp(-(x-1/fcs)*slp))

	cols = [[color[i]*(1-fc)/n*xf+ proto[i]*fc for i in [0,1,2]] for (fc, proto) in zip(f, protov)]
	avg = reduce(sum_rgb, cols)

	return avg

sigma = 8.0
def act(dist):
    return exp(sigma*(1-dist))
    
def fshift2(color, protov, distv):	
	n = len(protov)
	#f = [(cos((d**focus)*pi)/2+0.5)**slope for d in distv]
	a = map(act, distv)
	den = sum(a)
	f = [b / den for b in a]
	
	cols = [[color[i]*(1-fc)/n*xf+proto[i]*fc for i in [0,1,2]] for (fc, proto) in zip(f, protov)]
	avg = reduce(sum_rgb, cols)

	return avg

def xform(c):
	global cdist
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
	global mdist
	mdist = [0.0]*len(ppal)
	# TODO: use logger
	print 'processing...'
	tic = time()
	i2 = map(xform, idata)
	toc = time()
	dt = toc-tic
	print '...done in', dt, 'secs'
	newi.putdata(i2)
	if stats:
		mdist = [sd / len(idata) for sd in mdist]
		print 'mdist:', mdist
	return newi

#
# image analysis

from random import sample
def analyse(img, pal, k=1000):
	idata = img.getdata()
	smp = sample(idata, k)
	print 'processing...'
	tic = time()

	d = []
	for c in smp:
		# convert to float(0..1)
		v = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
		for p in pal:
			# calc distance
			dist = edist(v, p)
			d.append(dist)
	d = sorted(d)
	toc = time()
	dt = toc-tic
	print '...done in', dt, 'secs'
	perc = d[k/10]
	focus = -2.1356/log(perc)
	print 'perc={}, focus={}'.format(perc, focus)
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
		
def draw_palette(*pals):
	"""
	Draw palettes in pals onto one single
	Image.
	Usage: draw_palette(a,b,...)
	   or: draw_palette(*pals)
	Returns Image
	"""
	h = 50
	img = Image.new('RGB', (256,h*len(pals)))
	drw = ImageDraw.Draw(img)
	for p in range(len(pals)):
		cols = len(pals[p])
		w = 256 / cols
		for i in range(cols):
			x = i * w
			y = p * h
			pal = pals[p][i]
			col = (int(pal[0]*255.0), int(pal[1]*255.0), int(pal[2]*255.0))
			drw.rectangle([(x,y),(x+w,y+h)], fill=col)
	img.show()
	return img

