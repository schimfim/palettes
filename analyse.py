import hues
from genimg import gen_hs
from logging import info
import copy

# analysis params
nmap = 32
maxn = 0
focus = 0.4
limit = 0.0
dmax = 0.3
# training params
emph = 1.0
mu = 0.7
niter = 200
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

'''
Select sub-matrix from square matrix 'm' at
row/column i,j of size 3x3.
Border cases are either zero or column
wrapped if rot_j=True.
Returns tuple ([m[i-1][j-1]..m[i+1][j+1]],
               m[i][j])
'''
def select(m, i,j, rot_j=True):
	nr = len(m)
	nc = len(m[0])
	if nr != nc:
		raise ValueError('m must be square')
	vals = []
	for row in range(i-1,i+2):
		for col in range(j-1,j+2):
			if rot_j: col = col % nc
			if row<0 or row>=nr: val = 0.0
			elif col<0 or col>=nc: val=0.0
			else: val=m[row][col]
			if not(row==i and col==j):
				vals.append(val)
	return vals, m[i][j]

def extrema(hist):
	ext = copy.copy(hist)
	for i in range(len(hist)):
		for j in range(len(hist)):
			(vals, v) = select(hist, i,j)
			# edge detection
			sv = sorted(vals)
			if v - sum(sv[:4])/4 > dmax:
				ext[i][j] = 1.0
			else: 
				ext[i][j] = v**2
	return ext

'''
Return flattened list of new values from
vals matching frequency in freq.
vals must be sorted!
'''
def flatten(vals, freq):
	hist = sorted(freq)
	cum_dist = []
	sumf = 0
	for x in hist:
		sumf += x
		cum_dist.append(sumf)

	return cum_dist

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
			dhue = (hues[im1][j]-hue)*vals[im1][j] + (hues[i][jm1]-hue)*vals[i][jm1] + (hues[ip1][j]-hue)*vals[ip1][j] + (hues[i][jp1]-hue)*vals[i][jp1]

			new_hues[i][j] += (1-vals[i][j]) * dhue * mu
			sumd += dhue * mu * (1-vals[i][j])
			# sats
			jm1 = max(j - 1, 0)  # Bei saturation rotiert Spalte nicht!
			jp1 = min(j + 1, nmap-1)
			sat = sats[i][j]
			dsat = (sats[im1][j]-sat)*vals[im1][j] + (sats[i][jm1]-sat)*vals[i][jm1] + (sats[ip1][j]-sat)*vals[ip1][j] + (sats[i][jp1]-sat)*vals[i][jp1]

			new_sats[i][j] += dsat * mu * (1-vals[i][j])
			
	return new_hues, new_sats, sumd

def _xform(hsv, hues, sats, hilite=False):
	hn,sn,vn = hsv[0], hsv[1], hsv[2]

	# indices
	hi = int(round(hn*(nmap-1)))
	si = int(round(sn*(nmap-1)))
	hue = hues[hi][si]
	sat = sats[hi][si]
	
	if hilite:
		d = ((hue-hn)**2 + (sat-sn)**2)**0.5 / 2**0.5
		vn = d*nmap/4

	return (hue, sat, vn)

def _xform_vals(hsv, vals):
	hn,sn,vn = hsv[0], hsv[1], hsv[2]

	# indices
	hi = int(round(hn*(nmap-1)))
	si = int(round(sn*(nmap-1)))
	fq = vals[hi][si]
	
	
	if fq>limit:
		pass
		#sn *= fq
		#vn *= fq**2
	else: 
		sn = 0.0
		vn = 0.0
	

	return (hn, sn, vn)

def train_map(vals):
	(hues, sats) = gen_linmap(nmap)
	info('Start training')
	for i in range(niter):
		hues, sats, sumd = _update_xform(hues, sats, vals)
		if (i+1) % 10 == 0:
			info('Update iteration %d, %f', i+1, sumd)
		if abs(sumd) < float(dlimit)/nmap: break
	return hues, sats

def adapt(hsv, hues, sats, xfct=_xform, hilite=False):
	hsv_new = [xfct(c, hues, sats, hilite) for c in hsv]

	return hsv_new

def adapt_vals(hsv, vals):
	hsv_new = [_xform_vals(c, vals) for c in hsv]

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
		self.hsv_test = hues.PaletteTestBase.get_hsv('kueche')


	@unittest.skipUnless(test_all, 'test_all not set')
	def test_histogram(self):
		vals = create_histogram(self.hsv_data)
		img = gen_hs(vmap=vals, nc=nmap)
		img.show()

	@unittest.skipUnless(test_all, 'test_all not set')
	def test_extrema(self):
		vals = create_histogram(self.hsv_data)
		ext = extrema(vals)
		img = gen_hs(vmap=ext, nc=nmap)
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
		ext = extrema(vals)
		(hues, sats) = train_map(ext)

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
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_adapt_img(self):
		vals = create_histogram(self.hsv_data)
		ext = extrema(vals)
		(hues, sats) = train_map(ext)

		# show map
		img = gen_hs(hmap=hues, smap=sats, nc=nmap, v_def=1.0)
		img.show()

		# adapt test image
		hsv_new = adapt(self.hsv_test, hues, sats)

		self.render(hsv_new, palette=False)

	'''
	Foto modifiziert, Aenderungen hervorheben
	'''
	#@unittest.skipUnless(test_all, 'test_all not set')
	def test_highlight_changes(self):
		vals = create_histogram(self.hsv_data)
		ext = extrema(vals)
		(hues, sats) = train_map(ext)

		# show map
		img = gen_hs(hmap=hues, smap=sats, nc=nmap, v_def=1.0)
		img.show()

		# adapt test image
		hsv_new = adapt(self.hsv_test, hues, sats, hilite=True )

		self.render(hsv_new, palette=False)
		
		
	'''
	Foto mit Hauptfarben hervorgehoben
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_highlight_palette(self):
		vals = create_histogram(self.hsv_data)
		(hues, sats) = gen_linmap(nmap)

		# show map
		img = gen_hs(hmap=hues, smap=sats, nc=nmap, vmap=vals)
		img.show()

		# adapt self image
		hsv_new = adapt_vals(self.hsv_data, vals)
		self.render(hsv_new, palette=False)
		# adapt test image
		hsv_new = adapt_vals(self.hsv_test, vals)
		self.render(hsv_new, palette=False)


#
if __name__ == '__main__':
	unittest.main()

