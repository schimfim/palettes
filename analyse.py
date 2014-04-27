import hues
from genimg import gen_hs
from time import time
from logging import info

nmap=32
maxn = 0
focus = 0.4
mu = 0.7
niter = 50

def gen_linmap(n):
	ver = [[float(x)/n for x in range(n)] for y in range(n)]
	hor = [[float(y)/n for x in range(n)] for y in range(n)]
	return (hor, ver)

def _sample(hsv, hs_map):
	global maxn
	hn,sn,vn = hsv[0], hsv[1], hsv[2]
	
	# indices
	hi = int(hn*(nmap-1))
	si = int(sn*(nmap-1))
	hs_map[hi][si] += 1
	maxn = max(maxn, hs_map[hi][si])

def _update_xform(hues, sats, vals):
	sumd = 0.0
	for i in range(1,nmap-1):
		for j in range(1,nmap-1):
			# hues
			hue = hues[i][j]
			dhue = ((hues[i-1][j]-hue)* vals[i-1][j] + (hues[i][j-1]-hue)* vals[i][j-1] + (hues[i+1][j]-hue)* vals[i+1][j] + (hues[i][j+1]-hue)* vals[i][j+1]) * mu
			hues[i][j] += dhue
			sumd += dhue
			# sats
			sat = sats[i][j]
			dsat = ((sats[i-1][j]-sat)* vals[i-1][j] + (sats[i][j-1]-sat)* vals[i][j-1] + (sats[i+1][j]-sat)* vals[i+1][j] + (sats[i][j+1]-sat)* vals[i][j+1]) * mu
			sats[i][j] += dsat
			
	return sumd

def analyse(hsv):
	# setup map
	hs_map = [[0 for i in range(nmap)] for j in range(nmap)]
	
	# sample data
	for col in hsv:
		_sample(col, hs_map)
	info('Max n: %d', maxn)
	
	# generate map hsv values
	vals = [[(float(v)/maxn)**focus for v in row] for row in hs_map]
	#img = gen_hs(vmap=vals, nc=nmap)
	
	# calc xform matrix
	(hues, sats) = gen_linmap(nmap)
	for i in range(niter):
		sumd = _update_xform(hues, sats, vals)
		info('Update iteration %d, %f', i+1, sumd)
	
	# image xform matrix
	img = gen_hs(hmap=hues, smap=sats, nc=nmap, vmap=vals, v_def=1.0)
	
	return img

'''
Unit tests
'''
import unittest
test_all = False 

class TestColorAnalysis(hues.PaletteTestBase):
	
	# wie besser?
	@classmethod
	def setUpClass(cls):
		hues.PaletteTestBase.setUpClass()

	def applyFilter(self):
		print 'analysing ...'
		tic = time()
		img_new = analyse(self.hsv_data)
		toc = time()
		dt = toc-tic
		print '...done in {:.3f} secs'.format(dt)
		img_new.show()
		
	def setUp(self):
		img_key = 'herbst'
		self.img = hues.PaletteTestBase.imgs[img_key]
		self.hsv_data = hues.PaletteTestBase.hsv[img_key]
		self.filt = hues.Filter(8)

	#@unittest.skipUnless(test_all, 'test_all not set')
	def test_shift_hue(self):
		self.filt.hues[4] = 0.0
		self.filt.update()
		self.applyFilter()


#
if __name__ == '__main__':
	unittest.main()
	#suite = unittest.TestLoader().loadTestsFromTestCase(TestColorAnalysis)
	#unittest.TextTestRunner(verbosity = 1).run(suite)

