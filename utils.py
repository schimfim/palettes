import Image
import ImageDraw
from colorsys import hsv_to_rgb, rgb_to_hsv
import pdb
import genimg
from pals import pals

def load(filename, size=(256,256), thumbnail=True):
	img = Image.open(filename)
	if thumbnail:
		img.thumbnail(size, Image.ANTIALIAS)
	return img

def hues_to_rgb(hues):
	cols = [hsv_to_rgb(h, 1.0, 1.0) for h in hues]
	return cols

def rgb2hsv(data):
	'''
	Transform raw rgb data (bytes) to hsv
	floats.
	'''
	hsv = [rgb_to_hsv(r/255.0, g/255.0, b/255.0) for (r,g,b) in data]
	return hsv

def _draw_palette_with_hues(hues, h=20):
	"""
	Draw palettes with hue vector
	Returns Image
	"""
	width = 256
	n = len(hues)
	rgb = hues_to_rgb(hues)
	img = Image.new('RGB', (width, h))
	drw = ImageDraw.Draw(img)
	w = float(width) / n
	for i in range(n):
		x = i * w
		y = 0.0 * h
		pal = rgb[i]
		col = (int(pal[0]*255.0), int(pal[1]*255.0), int(pal[2]*255.0))
		drw.rectangle([(x,y),(x+w,y+h)], fill=col)
	
	return img

def add_palettes(img, hues, hues2=None, hues3=None):
	pal_img = _draw_palette_with_hues(hues, 10)
	img.paste(pal_img, (0,0))
	if hues2 !=None:
		pal_img = _draw_palette_with_hues(hues2, 10)
		img.paste(pal_img, (0,10))
		if hues3 !=None:
			pal_img = _draw_palette_with_hues(hues3, 10)
			img.paste(pal_img, (0,20))

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

def fromPalette(name):
		(h,s) = get_hues(name)
		filt = Filter(len(h))
		filt.hues = h
		filt.update()
		return filt


# ...
def clust1d(dat, n, nc=200):
	step = int(float(len(dat))/nc)
	sd = sorted(dat)
	c = []
	idx = 0
	
	# initialize with nc segments
	for i in range(nc):
		vals = sd[idx:idx+step-1]
		mean = sum(vals)/float(step)
		c.append({'mean':mean,'num':step})
		if i>0:
			diff = abs(c[i]['mean'] - c[i-1]['mean'])
			c[i-1]['diff'] = diff
		idx += step
	#pdb.set_trace()
	
	# merge segments until only n are left
	md = 0.0
	while nc > n:
	#while md<0.05:
		# find smallest difference
		md = 2.0
		mi = nc + 1
		for i in range(len(c)-1):
			if c[i]['diff'] < md:
				md = c[i]['diff']
				mi = i
		# merge segments mi and mi+1
		#print 'merge:', mi
		#pdb.set_trace()
		new_num = c[mi]['num'] + c[mi+1]['num']
		new_mean = (c[mi]['mean']*c[mi]['num'] + c[mi+1]['mean']*c[mi+1]['num']) / new_num
		c[mi]['num'] = new_num
		c[mi]['mean'] = new_mean
		c[mi+1:mi+2] = []
		nc -= 1
		# update diffs
		if mi<len(c)-2:
			new_diff = abs(c[mi]['mean']-c[mi+1]['mean'])
			c[mi]['diff'] = new_diff
		if mi>0:
			new_diff_left = abs(c[mi]['mean']-c[mi-1]['mean'])
			c[mi-1]['diff'] = new_diff_left
	print 'minimum distance:', md
	return [ci['mean'] for ci in c]

if __name__=='__main__':
	img = genimg.gen_3()
	img = load('orig/green.jpg')
	rgb_data = img.getdata()
	hsv_data = rgb2hsv(rgb_data)
	hues = clust1d([x[0] for x in hsv_data], 64, 64)
	
	#hues = [0.0, 0.2, 0.4, 0.6, 0.8]
	c_img = draw_palette_with_hues(hues)
	add_palette(img, c_img)
	img.show()

