import hues
from genimg import gen_hs
from logging import info
import copy

nmap = 16
maxn = 0
focus = 1.0
emph = 4.0
mu = 1.0
niter = 50
dlimit = 0

def gen_linmap(n):
	ver = [[float(x)/n for x in range(n)] for y in range(n)]
	hor = [[float(y)/n for x in range(n)] for y in range(n)]
	return (hor, ver)

def _sample(hsv, hs_map):
	global maxn
	hn,sn,vn = hsv[0], hsv[1], hsv[2]
	
	# indices
	hi = int(round(hn*(nmap-1)))
	si = int(round(sn*(nmap-1)))
	hs_map[hi][si] += 1
	maxn = max(maxn, hs_map[hi][si])

def create_histogram(hsv_data):
	# setup map
	hs_map = [[0 for i in range(nmap)] for j in range(nmap)]

	# sample data
	for col in hsv_data:
		_sample(col, hs_map)
	info('Max n: %d', maxn)

	# generate map hsv values
	vals = [[(float(v)/maxn)**focus for v in row] for row in hs_map]
	return vals

def act(diff):
	return ((diff + 1.0)/2.0)**emph

def _update_xform(hues, sats, vals):
	sumd = 0.0
	new_hues = copy.copy(hues)
	new_sats = copy.copy(sats)
	for i in range(nmap):
		for j in range(nmap):
			# hues
			hue = hues[i][j]
			val = vals[i][j]
			jm1 = (j - 1) % nmap # Spalte rotiert bei hue
			jp1 = (j + 1) % nmap
			im1 = max(i - 1, 0)
			ip1 = min(i + 1, nmap-1)
			dhue = (hues[im1][j]-hue)* act(vals[im1][j]-val) + (hues[i][jm1]-hue)* act(vals[i][jm1]-val) + (hues[ip1][j]-hue)* act(vals[ip1][j]-val) + (hues[i][jp1]-hue)* act(vals[i][jp1]-val)

			new_hues[i][j] += dhue * mu
			sumd += dhue * mu
			# sats
			jm1 = max(j - 1, 0)  # Bei saturation rotiert Spalte nicht!
			jp1 = min(j + 1, nmap-1)
			sat = sats[i][j]
			dsat = (sats[im1][j]-sat)* act(vals[im1][j]-val) + (sats[i][jm1]-sat)* act(vals[i][jm1]-val) + (sats[ip1][j]-sat)* act(vals[ip1][j]-val) + (sats[i][jp1]-sat)* act(vals[i][jp1]-val)

			new_sats[i][j] += dsat * mu
			
	return new_hues, new_sats, sumd

def _xform(hsv, hues, sats):
	hn,sn,vn = hsv[0], hsv[1], hsv[2]

	# indices
	hi = int(round(hn*(nmap-1)))
	si = int(round(sn*(nmap-1)))
	hue = hues[hi][si]
	sat = sats[hi][si]

	return (hue, sat, vn)

def train_map(vals):
	(hues, sats) = gen_linmap(nmap)
	for i in range(niter):
		hues, sats, sumd = _update_xform(hues, sats, vals)
		if (i+1) % 5 == 0:
			info('Update iteration %d, %f', i+1, sumd)
		if abs(sumd) < float(dlimit)/nmap: break
	return hues, sats

def adapt(hsv, hues, sats):
	hsv_new = [_xform(c, hues, sats) for c in hsv]
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
		
	def setUp(self):
		self.hsv_data = hues.PaletteTestBase.get_hsv('herbst')
		self.hsv_palette = hues.PaletteTestBase.get_hsv('palette')
		self.hsv_test = hues.PaletteTestBase.get_hsv('karussel')


	@unittest.skipUnless(test_all, 'test_all not set')
	def test_histogram(self):
		vals = create_histogram(self.hsv_data)
		img = gen_hs(vmap=vals, nc=nmap)
		img.show()

	@unittest.skipUnless(test_all, 'test_all not set')
	def test_xform(self):
		vals = create_histogram(self.hsv_data)
		# calc xform matrix
		(hues, sats) = train_map(vals)
		# visualize xform matrix
		img = gen_hs(hmap=hues, smap=sats, nc=nmap, v_def=1.0)
		img.show()
	'''
	Histogrammdarstellung sollte wie Original-Palette aussehen
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_identity_xform(self):
		focus = 0.4
		vals = create_histogram(self.hsv_palette)
		# calc xform matrix
		(hues, sats) = train_map(vals)
		# visualize xform matrix
		img = gen_hs(hmap=hues, smap=sats, vmap=vals, nc=nmap, v_def=1.0)
		img.show()

	'''
	Transformation mit Palette sollte Bild nicht veraendern
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_identity(self):
		vals = create_histogram(self.hsv_palette)
		(hues, sats) = train_map(vals)

		# adapt palette image
		hsv_new = adapt(self.hsv_palette, hues, sats)

		self.render(hsv_new, palette=False)

	'''
	Palette modifiziert
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_adapt_palette(self):
		vals = create_histogram(self.hsv_data)
		(hues, sats) = train_map(vals)

		# adapt palette image
		hsv_new = adapt(self.hsv_palette, hues, sats)

		self.render(hsv_new, palette=False)

	'''
	Foto mit eigenem Spektrum modifiziert.
	Sollte Farben hoechstens verstaerken
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_adapt_self(self):
		vals = create_histogram(self.hsv_data)
		(hues, sats) = train_map(vals)

		# adapt palette image
		hsv_new = adapt(self.hsv_data, hues, sats)

		self.render(hsv_new, palette=False)

	'''
	Foto modifiziert
	'''
	#@unittest.skipUnless(test_all, 'test_all not set')
	def test_adapt_img(self):
		vals = create_histogram(self.hsv_data)
		(hues, sats) = train_map(vals)

		# show map
		img = gen_hs(hmap=hues, smap=sats, nc=nmap, v_def=1.0)
		img.show()

		# adapt test image
		hsv_new = adapt(self.hsv_test, hues, sats)

		self.render(hsv_new, palette=False)


#
if __name__ == '__main__':
	unittest.main()

