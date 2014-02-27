from colorsys import rgb_to_hsv, hsv_to_rgb
from math import fmod, cos, pi

def rgb2hsv(data):
	'''
	Transform raw rgb data (bytes) to hsv
	floats.
	'''
	hsv = [rgb_to_hsv(r/255.0, g/255.0, b/255.0) for (r,g,b) in data]
	return hsv

def hsv2rgb(hsv):
	'''
	Transform hsv floats to raw rgb data
	(bytes)
	'''
	data = [hsv_to_rgb(h,s,v) for (h,s,v) in hsv]
	rgb = [(int(r*255.0), int(g*255.0), int(b*255.0)) for (r,g,b) in data]
	return rgb

focus = 1.7
def act(dist):
    f = [(cos((d**focus)*pi)/2+0.5)**5.0 for d in dist]
    return f

hues = [120.0/360, 0.0/360]
def rot(hsv):
	hn = hsv[0]
	dist = [abs(fmod(hn-hue, 1.0)) for hue in hues]
	f = act(dist)
	hn = sum([fc*hue+(1.0-fc)*hn for (fc,hue) in zip(f,hues)])/len(hues)
	return (hn, hsv[1], hsv[2])

def adapt(hsv):
	res = map(rot, hsv)
	return res
	
if __name__ == '__main__':
	import Image
	from palettes import load
	img = load('orig/kueche.jpg')
	rgb_data = img.getdata()
	hsv_data = rgb2hsv(rgb_data)
	
	angle = (200.0/360.0, 280.0/360.0)
	print 'mapping ...'
	hsv_new = adapt(hsv_data)
	print '... done'
	rgb_new = hsv2rgb(hsv_new)
	newi = img.copy()
	newi.putdata(rgb_new)
	
