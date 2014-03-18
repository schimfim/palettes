from colorsys import rgb_to_hsv, hsv_to_rgb
from math import fmod, cos, pi, log
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
	hh = sorted([hsv_to_rgb(h,s,v)[0] for (h,s,v) in p])
	ss = sorted([hsv_to_rgb(h,s,v)[1] for (h,s,v) in p])
	
	return (hh,ss)

def rdist(x,h):
	d = x - h
	if abs(d) > 0.5:
		d = d - d/abs(d)
	return d

def calc_focus(n):
	delta = 1.0/n/2
	gain = 0.5 # strenght of effect
	f = log(gain)/log(0.5*cos(2*pi*delta)+0.5)
	return f

def act(dist, focus):
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
		self.focus = 1.0
		self.update()
	
	def update(self):
		self.distm = [rdist(hm,hh) for (hm,hh) in zip(self.match, self.hues)]
		self.focus = calc_focus(self.order)
		print 'focus=', self.focus


def fromPalette(name):
		(h,s) = get_hues(name)
		filt = Filter(len(h))
		filt.hues = h
		#filt.sats = s
		filt.update()
		return filt


f_lin4 = Filter(4)

f_shift10 = Filter(4)
f_shift10.hues = [x+0.1 for x in f_shift10.match]
f_shift10.update()

f_mac = Filter(8)
f_mac.hues = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
f_mac.update()
print f_mac.match
print f_mac.distm
'''
h = get_hues('herbst')
f_herbst = Filter(len(h))
f_herbst.hues = h
f_herbst.update()
'''
f_pond = fromPalette('pond')

def rot(hsv, fdef):
	hn, sn = hsv[0], hsv[1]
	
	# calc hue
	dist = [rdist(hn,hue) for hue in fdef.match]
	f = act(dist, fdef.focus)
	hn -= sum([fc*d for (fc,d) in zip(f,fdef.distm)])
	
	if hn<0.0:
		hn += 1.0
	elif hn>1.0:
		hn -= 1.0
		
	if hn<0.0 or hn >1.0:
		raise ValueError('hn=%f' % hn)
	
	# calc saturation
	'''
	sf = sum(f)
	try:
		sn = sum([sn*fc*sc/sf for (sc,fc) in zip(fdef.sats,f)])
	except ZeroDivisionError:
		print 'f=', f
		raise 
	'''
	return (hn, hsv[1], hsv[2])
	
def adapt(hsv, filter):
	res = [rot(col, filter) for col in hsv]
	return res

def clust1d(dat, n):
	step = len(dat)/n
	sd = sorted(dat)
	c = sd[step::step]
	return c

def start():
	import Image
	from palettes import load
	from genimg import gen_hs
	'''
	print 'analyzing proto img ...'
	img_prot = load('orig/pond.jpg')
	rgb_prot = img_prot.getdata()
	hsv_prot = rgb2hsv(rgb_prot)
	nc = 5
	filt = Filter(nc)
	hc = clust1d([x[0] for x in hsv_prot], nc)
	hs = clust1d([x[1] for x in hsv_prot], nc)
	print len(hc)
	filt.hues = hc
	#filt.sats = hs
	filt.update()
	'''
	print 'loading input img ...'
	img = gen_hs(0.7)
	#img = load('orig/kueche.jpg')
	rgb_data = img.getdata()
	hsv_data = rgb2hsv(rgb_data)
	'''
	hm = clust1d([x[0] for x in hsv_data], nc)
	filt.match = hm
	filt.update()
	'''
	
	print 'mapping ...'
	tic = time()
	hsv_new = adapt(hsv_data, f_mac)
	toc = time()
	dt = toc-tic
	print '...done in {:.3f} secs'.format(dt)
	
	print 'rendering...'
	rgb_new = hsv2rgb(hsv_new)
	newi = img.copy()
	newi.putdata(rgb_new)
	newi.show()

if __name__ == '__main__':
	import cProfile
	#cProfile.run('start()')
	start()

