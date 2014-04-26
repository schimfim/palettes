import hues
from genimg import gen_hs
from time import time
from logging import info

nmap=32
maxn = 0
focus = 0.4

def gen_linmap(n):
	hor = [[float(x)/n for x in range(n)] for y in range(n)]
	ver = [[float(y)/n for x in range(n)] for y in range(n)]
	return (hor, ver)

def _sample(hsv, hs_map):
	global maxn
	hn,sn,vn = hsv[0], hsv[1], hsv[2]
	
	# indices
	hi = int(hn*(nmap-1))
	si = int(sn*(nmap-1))
	hs_map[hi][si] += 1
	maxn = max(maxn, hs_map[hi][si])
	
def analyse(hsv):
	# setup map
	hs_map = [[0 for i in range(nmap)] for j in range(nmap)]
	
	# sample data
	for col in hsv:
		_sample(col, hs_map)
	info('Max n: %d', maxn)
	
	# generate map hsv values
	vals = [[(float(v)/maxn)**focus for v in row] for row in hs_map]
	img = gen_hs(vmap=vals, nc=nmap)
	
	# calc xform matrix
	(hues, sats) = gen_linmap(nmap)
	for i in range(nmap):
		for j in range(nmap):
			pass 
	
	# 
	#xxx
	
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

