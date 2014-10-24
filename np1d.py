import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt

# params
nbins = 32
size = (512,512)
just_rgb = False
hcut = 0.0 # histogram cutoff min

def open_hsv(fname, size=(256,256)):
	img = Image.open(fname)
	img.thumbnail(size)
	ary = np.asarray(img)/255.0
	# xform to hsv
	if just_rgb:
		hsv = ary
	else: 
		hsv = rgb_to_hsv(ary)
	return hsv

def  rolling_window ( a ,  window ):
	shape  =  a . shape [: - 1 ]  +  ( a . shape [ - 1 ]  -  window  +  1 ,  window )
	strides  =  a . strides  +  ( a . strides [ - 1 ],)
	return  np . lib . stride_tricks . as_strided ( a ,  shape = shape ,  strides = strides )

def gen_map(vals, n=nbins, hue=False):
	# saturation
	fvals = vals.flatten()
	yh, xh, patches = plt.hist(fvals, bins=n, range=(0,1), normed=False , cumulative=False , histtype='step')
	if hue:
		# apply window
		M = 9
		win = np.kaiser(M, 3.0)
		yh = np.insert(yh, 0, np.zeros(M/2))
		yh = np.append(yh, np.zeros(M/2))
		yh = rolling_window(yh.T, M)
		yh = np.dot(yh, win)
	yh /= sum(yh)
	if hue:
		# adapted norm
		#yh = np.minimum(yh, hcut)
		yh[yh<=hcut] = 0
		yh /= sum(yh)
	yh = np.cumsum(yh)

	xhi = np.linspace(0,1,256)
	yhi = np.interp(xhi, yh, xh[1:])
	yhinv = np.interp(xhi, xh[1:], yh)
	#plt.plot(xhi, yhi)
	return (yhi, yhinv)
	
def apply_map(vals, map, bal=1.0, hue=False):
	hi = vals*255.0
	hi = hi.astype(np.uint8, copy=False)
	if not hue:
		vals_t = map[hi]*bal + vals*(1-bal)
	else: 
		mhi = map[hi] - 0.5
		#mhi[mhi>0.5] -= 1.0
		vhi = vals - 0.5
		#vhi[vhi>0.5] -= 1.0
		vals_t = mhi*bal + vhi*(1-bal) + 0.5
		#vals_t[vals_t<0] += 1.0
	return vals_t

def chain_maps(a, b):
	return (a + b) / 2.0
	# old:
	c = np.empty((3,256))
	for r in [0,1,2]:
		i = a[r] * 255.0
		i = i.astype(np.uint8, copy=False)
		c[r,:] = b[r,i]
	#c[1,0:50] = np.linspace(0,0.1,50)
	#c[1,:] = c[1,:] * np.power(np.linspace(0,1.0,256), 0.3)
	return c

def gen_linmaps():
	out = np.empty((3,256))
	out[0,:] = np.linspace(0,1,256)
	out[1,:] = np.linspace(0,1,256)
	out[2,:] = np.linspace(0,1,256)
	return out

def gen_maps(hsv):
	out = np.empty((3,256))
	inv = np.empty((3,256))
	# gen output arrays
	(out[0,:], inv[0,:]) = gen_map(hsv[...,0].flatten(), hue=True)
	(out[1,:], inv[1,:]) = gen_map(hsv[...,1].flatten())
	(out[2,:], inv[2,:]) = gen_map(hsv[...,2].flatten())
	return (out, inv)

def apply_maps(out, maps, bal=(1.0,1.0,1.0)):
	res = np.empty_like(out)
	res[...,0] = apply_map(out[...,0], maps[0], bal[0], hue=not just_rgb)
	res[...,1] = apply_map(out[...,1], maps[1], bal[1])
	res[...,2] = apply_map(out[...,2], maps[2], bal[2])
	return res

if __name__ == '__main__':
	# input image
	hsv = open_hsv('orig/dunes.jpg')
	# output image
	out_hsv = open_hsv('orig/karussel.jpg', size=size)

	(maps, imaps) = gen_maps(hsv)
	#plt.show()
	plt.clf()
	
	(out_maps, iout_maps) = gen_maps(out_hsv)
	#plt.show()
	plt.clf()
	
	xm = gen_linmaps()
	x = xm[0]
	#bal=(0.6,0.8,1.0)
	bal=(1.0,1.0,1.0)

	#maps2 = chain_maps(out_maps, maps)
	maps2 = maps
	# color cycle: bgr = hsv
	plt.plot(x, maps2.T)
	plt.show()
	
	hsv_out = apply_maps(out_hsv, maps2, bal=bal)
	if just_rgb:
		rgb = hsv_out * 255
	else: 
		rgb = hsv_to_rgb(hsv_out)*255
		hsv = hsv_out * 255
	nimg = Image.fromarray(rgb.astype(np.uint8))
	hsvimg = Image.fromarray(hsv.astype(np.uint8))
	nimg.show()
	
	'''
	# composite
	import ui
	
	isize = nimg.size
	nimg.save('tmp.jpg')
	uiimg = ui.Image.named('tmp.jpg')
	orig = ui.Image.named('orig/dunes.jpg')
	with ui.ImageContext(*isize) as ctx:
		orig.draw(0,0,*isize)
		ui.set_blend_mode(ui.BLEND_MULTIPLY)
		uiimg.draw(0,0,200,200)
		ui.set_blend_mode(ui.BLEND_COLOR)
		ui.fill_rect(0, 0, 100, 100)
		cimg = ctx.get_image()
		cimg.show()
'''
