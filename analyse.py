import hues
from genimg import gen_hs
from logging import info
import copy
from math import sqrt
from pdb import set_trace

# analysis params
nmap = 16
maxn = 0
focus = 0.8
limit = 0.0
dmax = 0.3
# training params
emph = 1.0
mu = 0.8
niter = 500
dlimit = 0
# other parms
logstep = 50
contrast = 3.0


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

'''
return histogram with extrema set to 1.0.
If cents=true, also returns centroids as
arrays of hues and saturations plus array
of centroid weights.
NOTE: fuer filter wird nur huec/satc benoetigt
'''
def extrema(hist, cents=False):
	ext = copy.copy(hist)
	(hues, sats) = gen_linmap(nmap)
	huec, satc, wc = [], [], []
	for i in range(len(hist)):
		for j in range(len(hist)):
			(vals, v) = select(hist, i,j)
			# edge detection
			sv = sorted(vals)
			if v - sum(sv[:4])/4 > dmax:
				wc.append(v)
				huec.append(hues[i][j])
				satc.append(sats[i][j])
				ext[i][j] = 1.0
			else: 
				ext[i][j] = 0.0
				
	if cents:
		return ext, huec, satc, wc
	else: 
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
			#dhue = (hues[im1][j]-hue)*vals[im1][j] + (hues[i][jm1]-hue)*vals[i][jm1] + (hues[ip1][j]-hue)*vals[ip1][j] + (hues[i][jp1]-hue)*vals[i][jp1]
			dhue = (hues[im1][j]-hue)*vals[im1][j]**emph + (hues[i][jm1]-hue)*vals[i][jm1]**emph + (hues[ip1][j]-hue)*vals[ip1][j]**emph + (hues[i][jp1]-hue)*vals[i][jp1]**emph
			
			new_hues[i][j] += (1-vals[i][j]) * dhue * mu
			sumd += dhue * mu * (1-vals[i][j])
			# sats
			jm1 = max(j - 1, 0)  # Bei saturation rotiert Spalte nicht!
			jp1 = min(j + 1, nmap-1)
			sat = sats[i][j]
			#dsat = (sats[im1][j]-sat)*vals[im1][j] + (sats[i][jm1]-sat)*vals[i][jm1] + (sats[ip1][j]-sat)*vals[ip1][j] + (sats[i][jp1]-sat)*vals[i][jp1]
			dsat = (sats[im1][j]-sat)*vals[im1][j]**emph + (sats[i][jm1]-sat)*vals[i][jm1]**emph + (sats[ip1][j]-sat)*vals[ip1][j]**emph + (sats[i][jp1]-sat)*vals[i][jp1]**emph
			
			new_sats[i][j] += dsat * mu * (1-vals[i][j])
			
	return new_hues, new_sats, sumd

def _update_xform_max(hues, sats, vals):
	sumd = 0.0
	new_hues = copy.copy(hues)
	new_sats = copy.copy(sats)
	for i in range(nmap):
		for j in range(nmap):
			(surr_vals, val) = select(vals, i,j)
			if val==1.0: continue
			max_v = max(surr_vals)
			idx_v = surr_vals.index(max_v)
			(surr_hues, hue) = select(hues, i,j)	
			(surr_sats, sat) = select(sats, i,j, rot_j=False)	
			mhue = surr_hues[idx_v]
			msat = surr_sats[idx_v]
			dhue = (mhue-hue)*max_v
			dsat = (msat-sat)*max_v

			new_hues[i][j] += (1-val) * dhue * mu
			new_sats[i][j] += (1-val) * dsat * mu
			sumd += dhue * mu * (1-val)
			
	return new_hues, new_sats, sumd

def rdist(x,h):
	d = x - h
	if abs(d) > 0.5:
		d = d - d/abs(d)
	return d
	
def memb(hue,sat, huec,satc):
	hdist = [rdist(hue,c) for c in huec]
	sdist = [sat-c for c in satc]
	dist = [rdist(hue,c)**2+(sat-s)**2 for (c,s) in zip(huec,satc)]

	u = [1/sum([((di+0.0001)/(dj+0.0001))**(2.0/(contrast-1.0)) for dj in dist]) for di in dist]
	
	return u, hdist, sdist

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

def train_map(vals, updt_fct=_update_xform):
	(hues, sats) = gen_linmap(nmap)
	info('Start training')
	for i in range(niter):
		hues, sats, sumd = updt_fct(hues, sats, vals)
		if (i+1) % logstep == 0:
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
		self.hsv_data = hues.PaletteTestBase.get_hsv('pond')
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
		ext = extrema(vals)
		img0 = gen_hs(vmap=ext, nc=nmap)
		
		# calc xform matrix
		(hues, sats) = train_map(ext)
		# visualize xform matrix
		img = gen_hs(hmap=hues, smap=sats, nc=nmap, v_def=1.0)
		img0.show()
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
	Foto modifiziert mit max training algo
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_adapt_img_max(self):
		vals = create_histogram(self.hsv_data)
		ext = extrema(vals)
		(hues, sats) = train_map(ext, updt_fct=_update_xform_max)

		# show map
		img = gen_hs(hmap=hues, smap=sats, nc=nmap, v_def=1.0)
		img.show()

		# adapt test image
		hsv_new = adapt(self.hsv_test, hues, sats)

		self.render(hsv_new, palette=False)

	'''
	Foto modifiziert, Aenderungen hervorheben
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
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

	'''
	Test activation function
	'''
	@unittest.skipUnless(test_all, 'test_all not set')
	def test_memb(self):
		hue = 0.3
		sat = 0.6
		huec = [0.0, 0.5, 0.7]
		satc = [0.1, 0.2, 0.3]
		u, d = memb(hue,sat,huec,satc)
		print 
		print u
		print d
		self.assertEqual(sum(u), 1.0)
		

	'''
	Foto modifiziert mit cluster membership
	'''
	#@unittest.skipUnless(test_all, 'test_all not set')
	def test_adapt_img_memb(self):
		vals = create_histogram(self.hsv_data)
		(ext, huec, satc, wc) = extrema(vals, cents=True)

		# show extrema
		img = gen_hs(vmap=ext, nc=nmap)
		img.show()

		# adapt test image
		hsv_new = []
		for hsv in self.hsv_test:
			hue = hsv[0]
			sat = hsv[1]
			u,hdist,sdist = memb(hue,sat,huec,satc)
			dhue = sum([d*m for (d,m) in zip(hdist,u)])
			dsat = sum([d*m for (d,m) in zip(sdist,u)])
			hue -= dhue
			sat -= dsat
			hsv_new.append([hue, sat, hsv[2]])

		self.render(hsv_new, palette=False)

#
if __name__ == '__main__':
	unittest.main()

