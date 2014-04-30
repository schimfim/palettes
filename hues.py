from colorsys import rgb_to_hsv, hsv_to_rgb
from math import fmod, cos, pi, log
from pals import pals
from time import time
from pdb import set_trace
from  utils import add_palettes, clust1d, load, rgb2hsv, hsv2rgb, get_hues
import logging
logging.basicConfig(level=logging.INFO)

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
		self.highlight = False 
		self.gain = 0.5
		self.act_fcn = act

		self.update()

	def update_simple(self):
		self.distm = [rdist(hm,hh) for (hm,hh) in zip(self.match, self.hues)]
		self.focus = self.calc_focus()

	def update(self, new_hues=None):
		n = self.order
		self.distm = [0.0]*n
		self.focus = self.calc_focus()
		if not new_hues:
			new_hues = self.hues
		for (i,hm) in enumerate(self.match):
			dist = [rdist(hm,hh) for hh in new_hues]
			f = self.act_fcn(dist, self.focus)
			self.distm[i] = sum([fc*d for (fc,d) in zip(f,dist)])/sum(f)

	def calc_focus(self):
		delta = 1.0/self.order/2
		f = log(self.gain)/log(0.5*cos(2*pi*delta)+0.5)
		return f
		
	def set_switches(self, *idx):
		self.switch = [0.0]*self.order
		for i in idx:
			self.switch[i] = 1.0

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
# only useful for testing, errors due to
# multiplication with distm!
def act_max(dist, focus):
	md = min([abs(d) for d in dist])
	f = []
	for d in dist:
		if abs(d) == md: f.append(1.0)
		else: f.append(0.0)
	return f

def rot(hsv, fdef):
	hn,sn,vn = hsv[0], hsv[1], hsv[2]
	dist = [rdist(hn,hue) for hue in fdef.match]
	f = fdef.act_fcn(dist, fdef.focus)
	hn -= sum([fc*d for (fc,d) in zip(f,fdef.distm)])
	if fdef.highlight:
		md = max([abs(d) for d in fdef.distm])
		lim = md * 0.7
		vf = sum([fc for (fc,d) in zip(f,fdef.distm) if abs(d) >= lim])
		vn *= vf

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
from random import random
from Image import new
test_all = False

class PaletteTestBase(unittest.TestCase):
	@classmethod
	def load(cls, key):
		img = load('orig/{}.jpg'.format(key))
		cls.imgs[key] = img
		cls.hsv[key] = rgb2hsv(img.getdata())
	
	@classmethod
	def get_hsv(cls, key):
		if key not in cls.hsv:
			cls.load(key)
		return cls.hsv[key]
	
	@classmethod
	def setUpClass(cls):
		# store test images in dict
		cls.imgs = {}
		cls.hsv = {}
		print 
		logging.info('Init images')
		# palette
		img = gen_hs(1.0)
		hsv_data = rgb2hsv(img.getdata())
		cls.imgs['palette'] = img
		cls.hsv['palette'] = hsv_data
		# load images
		#cls.load('herbst')
		#cls.load('kueche')
		#cls.load('orig_08')
		#cls.load('karussel')
		#cls.load('pond')
		#cls.load('city')

	def applyFilter(self):
		print 'mapping ...'
		tic = time()
		hsv_new = adapt(self.hsv_data, self.filt)
		toc = time()
		dt = toc-tic
		print '...done in {:.3f} secs'.format(dt)
		self.render(hsv_new)
		
	def render(self, hsv_new, palette = True):
		print 'rendering...'
		rgb_new = hsv2rgb(hsv_new)
		# newi = self.img.copy() NEIN: behaelt alte Daten!
		newi = new('RGB', (256,256))
		newi.putdata(rgb_new)
		if palette:
			dists = [h-d for (h,d) in zip(self.filt.match, self.filt.distm)]
			add_palettes(newi, self.filt.match, self.filt.hues, dists)
		newi.show()


class TestColorMods(PaletteTestBase):
	def setUp(self):
		img_key = 'herbst'
		self.img = TestColorMods.imgs[img_key]
		self.hsv_data = TestColorMods.hsv[img_key]
		self.filt = Filter(8)

	@unittest.skipUnless(test_all, 'test_all not set')
	def test_shift_hue(self):
		self.filt.hues[4] = 0.0
		self.filt.update()
		self.applyFilter()
	
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_shift_match(self):
		self.filt.match[4] += 0.1
		self.filt.update()
		self.applyFilter()
		
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_shift_dist(self):
		self.filt.distm = [0.0]*self.filt.order
		self.filt.distm[4] = 0.2
		self.filt.focus = 4.0
		self.applyFilter()
	
	'''
	Test highlighting of a single color
	(set saturation proportional to
	membership)
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_highlight_hue(self):
		self.filt.set_switches(2)
		self.filt.desat = True
		self.applyFilter()
	
	'''
	Test shifting selected colors
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_shift_select(self):
		self.filt.set_switches(2,7)
		self.filt.hues[7] = 0.3
		self.filt.hues[2] = 0.0
		self.filt.update()
		self.filt.desat = True 
		self.applyFilter()
	
	'''
	Identity with highlight
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_identity(self):
		self.filt.focus = 30.0
		self.filt.desat = True 
		self.applyFilter()
	
	'''
	Test pre defined palette.
	Linspace = [0.0611, 0.1861, 0.3111, 0.4361, 0.5611, 0.6861, 0.8111, 0.9361]
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_palette(self):
		self.filt.hues = [0.1, 0.15, 0.3111, 0.4361, 0.5, 0.6861, 0.8111, 0.04]
		self.filt.update()
		self.applyFilter()

	'''
	Test delta highlighting.
	All dists below limit will be darkened
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_delta_highlight(self):
		self.filt.hues = [0.1, 0.15, 0.3111, 0.4361, 0.5, 0.6861, 0.8111, 0.04]
		self.filt.update()
		logging.info('Set distm to %s', self.filt.distm)
		self.filt.highlight = True 
		self.applyFilter()

	'''
	Test delta setting with single color
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_delta_set_single(self):
		self.filt.hues = [0.43]
		self.filt.update()
		logging.info('Set distm to %s', self.filt.distm)
		self.filt.highlight = True 
		self.applyFilter()

	'''
	Test delta setting with fewer colors.
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_delta_setting(self):
		self.filt.hues = [0.0, 1.0/3, 2.0/3]
		self.filt.update()
		logging.info('Set distm to %s', self.filt.distm)
		self.applyFilter()

	'''
	Test delta setting with palette.
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_delta_set_palette(self):
		(self.filt.hues,foo) = get_hues('rgb')
		self.filt.update()
		logging.info('Set distm to %s', self.filt.distm)
		self.filt.highlight = False 
		self.applyFilter()

	'''
	Test delta setting with randon palette.
	'''
	#@unittest.skipUnless(test_all, 'test_all not set')
	def test_delta_set_random(self):
		self.filt.hues = sorted([random() for i in range(5)])
		self.filt.update()
		logging.info('Set distm to %s', self.filt.distm)
		self.filt.highlight = False 
		self.applyFilter()

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

if __name__ == '__main__':
	unittest.main()

