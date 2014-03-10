from colorsys import rgb_to_hsv, hsv_to_rgb
from math import fmod, cos, pi
from pals import pals
from time import time

def rgb2hsv(data):
	'''
	Transform raw rgb data (bytes) to hsv
	floats.
	'''
	hsv = [rgb_to_hsv(r/255.0, g/255.0, b/255.0) for (r,g,b) in data]
	return hsv

def lim(v):
	if abs(v) > 1:
		return 1.0
	elif abs(v) <0:
		return 0.0
	else: 
		return v

def h2r(h,s,v):
	return hsv_to_rgb(lim(h),lim(s),lim(v))

def hsv2rgb(hsv):
	'''
	Transform hsv floats to raw rgb data
	(bytes)
	'''
	data = [hsv_to_rgb(h,s,v) for (h,s,v) in hsv]
	rgb = [(int(r*255.0), int(g*255.0), int(b*255.0)) for (r,g,b) in data]
	return rgb

def get_hues(name):
	p = pals[name]
	h = sorted([hsv_to_rgb(h,s,v)[0] for (h,s,v) in p])
	return h

def rdist(x,h):
	d = x - h
	if abs(d) > 0.5:
		d = d - d/abs(d)
	return d

focus = 4.0
def act(dist):
	f = [(cos(d*2*pi)/2+0.5)**focus for d in dist]
	return f

class Filter(object):
	def __init__(self, ord=4):
		self.order = ord
		# by default init linear filter
		linrange = range(0,360,360/ord)
		self.match = [c/360.0 for c in linrange]
		self.hues = [c/360.0 for c in linrange]
		self.sats = [1.0] * ord
		self.update()
	
	def update(self):
		self.distm = [rdist(hm,hh) for (hm,hh) in zip(self.match, self.hues)]
		
f_lin4 = Filter(4)

f_shift10 = Filter(4)
f_shift10.hues = [x+0.1 for x in f_shift10.match]
f_shift10.update()

f_satred = Filter(4)
f_satred.sats[0] = 2.0

h = get_hues('herbst')
f_herbst = Filter(len(h))
f_herbst.hues = h
f_herbst.update()

h = get_hues('pond')
f_pond = Filter(len(h))
f_pond.hues = h
f_pond.update()

def rot(hsv, fdef):
	hn, sn = hsv[0], hsv[1]
	
	# calc hue
	dist = [rdist(hn,hue) for hue in fdef.match]
	f = act(dist)
	hn -= sum([fc*d for (fc,d) in zip(f,fdef.distm)])
		
	# calc saturation
	sf = sum(f)
	try:
		sn = sum([sn*fc*sc/sf for (sc,fc) in zip(fdef.sats,f)])
	except ZeroDivisionError:
		print 'f=', f
		raise 
	
	return (hn, sn, hsv[2])
	
def adapt(hsv, filter):
	res = [rot(col, filter) for col in hsv]
	return res
	
if __name__ == '__main__':
	import Image
	from palettes import load
	from genimg import gen_hs

	print 'loading ...'
	img = gen_hs(0.7)
	img = load('orig/kueche.jpg')
	rgb_data = img.getdata()
	hsv_data = rgb2hsv(rgb_data)

	print 'mapping ...'
	tic = time()
	hsv_new = adapt(hsv_data, f_pond)
	toc = time()
	dt = toc-tic
	print '...done in {:.3f} secs'.format(dt)
	
	print 'rendering...'
	rgb_new = hsv2rgb(hsv_new)
	newi = img.copy()
	newi.putdata(rgb_new)
	newi.show()
	
