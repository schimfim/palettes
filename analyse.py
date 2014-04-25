import hues

nmap=32

def _sample(hsv):
	hn,sn,vn = hsv[0], hsv[1], hsv[2]
	
	# indices
	hi = int(hn*(nmap-1))
	si = int(sn*(nmap-1))
	
def analyse(hsv, hsmap):
	# setup map
	hs_map = [[0.0 for i in range(nmap)] for j in range(nmap)]]
	
	# sample data
	for col in hsv:
		_sample(col, hs_map)
	
	# generate map hsv values
	
	
	return hsv_new

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
		hsv_new = analyse(self.hsv_data)
		toc = time()
		dt = toc-tic
		print '...done in {:.3f} secs'.format(dt)
		self.render(hsv_new)
		
	def render(self, hsv_new):
		print 'rendering...'
		rgb_new = hsv2rgb(hsv_new)
		newi = self.img.copy()
		newi.putdata(rgb_new)
		newi.show()

	def setUp(self):
		img_key = 'karussel'
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

