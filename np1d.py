import numpy as np, Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import  matplotlib.pyplot as plt

# params
nbins = 64
size = (512,512)

def open_hsv(fname, size=(256,256)):
	img = Image.open(fname)
	img.thumbnail(size)
	ary = np.asarray(img)/255.0
	# xform to hsv
	hsv = rgb_to_hsv(ary)
	#hsv = ary
	return hsv

def gen_map(vals, n=nbins):
	# process hues
	yh, xh, patches = plt.hist(vals.flatten(), bins=n, range=(0,1), normed=True, cumulative=True, histtype='step')
	xhi = np.linspace(0,1,256)
	yhi = np.interp(xhi, yh, xh[1:])
	#plt.plot(xhi, yhi)
	return yhi
	
def apply_map(vals, map, bal=1.0):
	hi = vals*255.0
	hi = hi.astype(np.uint8, copy=False)
	vals_t = map[hi]*bal + vals*(1-bal)
	return vals_t

def gen_linmaps():
	out = np.empty((3,256))
	out[0,:] = np.linspace(0,1,256)
	out[1,:] = np.linspace(0,1,256)
	out[2,:] = np.linspace(0,1,256)
	return out

def gen_maps(hsv):
	out = np.empty((3,256))
	# gen output arrays
	out[0,:] = gen_map(hsv[...,0].flatten())
	out[1,:] = gen_map(hsv[...,1].flatten())
	out[2,:] = gen_map(hsv[...,2].flatten())
	return out

def apply_maps(out, maps, bal=1.0):
	res = np.empty_like(out)
	res[...,0] = apply_map(out[...,0], maps[0], bal)
	res[...,1] = apply_map(out[...,1], maps[1], bal)
	res[...,2] = apply_map(out[...,2], maps[2], bal)
	return res

if __name__ == '__main__':
	# input image
	hsv = open_hsv('orig/pond.jpg')
	# output image
	out_hsv = open_hsv('orig/kueche.jpg', size=size)

	maps = gen_maps(hsv)
	plt.show()
	plt.clf()
	
	out_maps = gen_maps(out_hsv)
	plt.show()
	plt.clf()
	
	xm = gen_linmaps()
	x = xm[0]
	#wgt = np.abs(maps - out_maps)
	#diff_map = gen_linmaps()*(1-wgt) + (wgt * maps)
	#diff_map = np.abs(maps - gen_linmaps()) / gen_linmaps()
	diff_map = (maps - gen_linmaps())
	#diff_map = np.power(diff_map, 2)
	mx = np.max(diff_map)
	bal = 1.0
	#diff_map = np.clip(diff_map,0,1)
	# color cycle: bgr = hsv
	plt.plot(x, maps.T)
	plt.show()
	hsv_out = apply_maps(out_hsv, maps, bal=bal)

	rgb = hsv_to_rgb(hsv_out)*255
	#rgb = hsv_out*255
	nimg = Image.fromarray(rgb.astype(np.uint8))
	nimg.show()

