from colorsys import rgb_to_hsv, hsv_to_rgb
from math import fmod, cos, pi, log
from pals import pals
from time import time
import pdb
from  utils import add_palettes, clust1d, load, rgb2hsv, hsv2rgb

class Filter(object):
	def __init__(self, ord=4):
		self.order = ord
		# by default init linear filter
		linrange = range(180/ord,360,360/ord)
		self.match = [c/360.0 for c in linrange]
		self.hues = [c/360.0 for c in linrange]
		self.switch = [1.0] * ord
		self.sats = [1.0] * ord
		self.focus = 1.0

		# configuration
		self.desat = False
		self.hue_mode = False
		self.gain = 0.5
		self.act_fcn = act

		self.update()

	def update(self):
		self.distm = [rdist(hm,hh) for (hm,hh) in zip(self.match, self.hues)]
		self.focus = self.calc_focus()

	def calc_focus(self):
		delta = 1.0/self.order/2
		f = log(self.gain)/log(0.5*cos(2*pi*delta)+0.5)
		return f


def rdist(x,h):
	d = x - h
	if abs(d) > 0.5:
		d = d - d/abs(d)
	return d


### Activation functions ###

def act(dist, focus):
	f = [(cos(d*2*pi)/2+0.5)**focus for d in dist]
	return f

''' 1.0 for min distance, 0.0 otherwise '''
def act_max(dist, focus):
	md = min(dist)
	f = []
	for d in dist:
		if d == md: f.append(1.0)
		else: f.append(0.0)
	return f

def rot(hsv, fdef):
	hn,sn,vn = hsv[0], hsv[1], hsv[2]
	dist = [rdist(hn,hue) for hue in fdef.match]
	f = fdef.act_fcn(dist, fdef.focus)
	if not fdef.hue_mode:
		hn -= sum([fc*d for (fc,d) in zip(f,fdef.distm)])
	else: 
		hn = sum([fc*d for (fc,d) in zip(f,fdef.hues)]) / fdef.order

	if hn<0.0: hn += 1.0
	elif hn>1.0: hn -= 1.0
	if hn<0.0 or hn >1.0:
		raise ValueError('hn=%f' % hn)
	
	if fdef.desat:
		sf = sum([fi for (fi,di) in zip(f,fdef.switch) if di != 0.0])
		#sf = sum(f)/fdef.order
		sn *= sf
	
	return (hn,sn,vn)
	
def adapt(hsv, filter):
	res = [rot(col, filter) for col in hsv]
	return res


'''
Unit tests
'''
import unittest
from genimg import gen_hs
from utils import load
test_all = False

class TestColorMods(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		# store test images in dict
		cls.imgs = {}
		cls.hsv = {}
		#
		img = gen_hs(1.0)
		hsv_data = rgb2hsv(img.getdata())
		cls.imgs['palette'] = img
		cls.hsv['palette'] = hsv_data
		#
		img = load('orig/herbst.jpg')
		cls.imgs['herbst'] = img
		cls.hsv['herbst'] = rgb2hsv(img.getdata())
	
	def setUp(self):
		self.img = TestColorMods.imgs['palette']
		self.hsv_data = TestColorMods.hsv['palette']
		self.filt = Filter(8)

	@unittest.skipUnless(test_all, 'test_all not set')
	def test_blu2red(self):
		self.filt.hues[4] = 0.0
		self.filt.update()
		self.applyFilter()
	
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_blu_shift(self):
		self.filt.match[4] += 0.1
		self.filt.update()
		self.applyFilter()
		
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_single_hue(self):
		self.filt.distm = [0.0]*self.filt.order
		self.filt.distm[4] = 0.2
		self.filt.focus = 4.0
		self.applyFilter()
	
	'''
	Test highlighting of a single color
	(set brightness proportional to
	membership)
	'''
	#@unittest.skipUnless(test_all, 'test_all not set')
	def test_highlight_hue(self):
		n = self.filt.order
		hsv_data = TestColorMods.hsv['herbst']
		self.filt.switch = [0.0]*n
		self.filt.switch[1] = 1.0
		self.filt.desat = True
		hsv_new = adapt(hsv_data, self.filt)
		self.render(hsv_new)
	
	'''
	Test shifting selected colors
	(set brightness proportional to
	membership)
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_shift_select(self):
		n = self.filt.order
		hsv_data = TestColorMods.hsv['herbst']
		self.filt.distm = [0.0]*n
		self.filt.distm[7] = 0.3
		self.filt.distm[2] = -0.4
		self.filt.desat = True 
		hsv_new = adapt(hsv_data, self.filt)
		self.render(hsv_new)
	
	'''
	Same as before but not with shifting
	but setting absolute colors
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_set_select(self):
		n = self.filt.order
		hsv_data = TestColorMods.hsv['herbst']
		hsv_new = []
		self.filt.distm = [0.0]*n
		self.filt.distm[7] = 0.3
		self.filt.distm[2] = -0.4
		for (hn,sn,vn) in hsv_data:
			dist = [rdist(hn,hue) for hue in self.filt.match]
			f = act(dist, self.filt.focus)
			hn = sum([fc*d for (fc,d) in zip(f,self.filt.hues)])
			sf = sum([fi for (fi,di) in zip(f,self.filt.distm) if di != 0.0])
			sn *= sf
			hsv_new.append((hn,sn,vn))
		self.render(hsv_new)

	'''
	Identity with highlight
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_identity(self):
		hsv_data = TestColorMods.hsv['herbst']
		n = self.filt.order
		self.filt.distm = [0.1]*n
		#self.filt.focus = 50.0
		self.filt.desat = True 
		self.filt.hue_mode = True
		hsv_new = adapt(hsv_data, self.filt)
		self.render(hsv_new)
	

	# material not ready
	@unittest.skip("not complete")
	def test_imageMatch(self):
		img_prot = load('orig/kueche.jpg')
		rgb_prot = img_prot.getdata()
		hsv_prot = rgb2hsv(rgb_prot)
		nc = 16
		filt = Filter(nc)
		hc = clust1d([x[0] for x in hsv_prot], nc, 1000)
		add_palettes(img_prot, hc)
		img_prot.show()
		filt.hues = hc
		filt.update()
		
	'''
	Test utilities
	'''
	def applyFilter(self):
		print 'mapping ...'
		tic = time()
		hsv_new = adapt(self.hsv_data, self.filt)
		toc = time()
		dt = toc-tic
		print '...done in {:.3f} secs'.format(dt)
		self.render(hsv_new)
		
	def render(self, hsv_new):
		print 'rendering...'
		rgb_new = hsv2rgb(hsv_new)
		newi = self.img.copy()
		newi.putdata(rgb_new)
		add_palettes(newi, self.filt.match, self.filt.hues)
		newi.show()
	

if __name__ == '__main__':
	unittest.main()

