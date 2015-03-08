# coding: utf-8
# todo:
#

import ui
import photos
import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
#import npclust3d
import locals3d
import logging as log

#nptest.plot = False  

thbsize = (256,256)
hdsize = (1024,1024)

# model
class Model(object):
	pass
global md
md = Model()

# interface
# todo: als "echtes" Interface bauen
locals3d.verbose = False 
def calcParams(filt,img=None):
	params = {}
	params['filt'] = locals3d.find_peaks(filt)[0]
	if img != None:
		params['img'] = locals3d.find_peaks(img)
	return params
def applyFilter(a,p, controls):
	ct = controls['mu0']
	if 'img' in p:
		pass
	else: 
		out = locals3d.fastApplyCents(a, p['filt'], mu0=ct, CT=3.0)
	return out

def uiimg_from_array(ary):
	# ary in (0,1)
	int_ary = ary*255
	int_ary = int_ary.astype(np.uint8)
	plt.imsave('tmp.png', int_ary, format='png')
	uimg = ui.Image.named('tmp.png')
	return uimg

# open image
@ui.in_background
def open_action(sender, img=None):
	global md
	if img is None:
		img = photos.pick_image(show_albums=True)
	md.hd_img = img.copy()
	md.hd_img.thumbnail(hdsize)
	img.thumbnail(thbsize)
	md.orig_ary = np.asarray(img)/255.0
	md.orig_img = uiimg_from_array(md.orig_ary)
	# update ui
	img_view = v['theImage']
	img_view.image = md.orig_img

def pick_image():
	img = photos.pick_image(show_albums=True)
	img.thumbnail((256,256))
	return img

# add filter
@ui.in_background
def add_filter(sender, img=None ):
	if img is None :
		img = pick_image()
	else: 
		img.thumbnail((256,256))
	ary = np.asarray(img)/255.0
	md.filt_ary = ary
	uiimg = uiimg_from_array(ary)
	# calc cube
	#(md.cube, nc) = npclust3d.calcCube(ary, md.h_perc, md.nbrs, md.orig, md.contr)
	md.cube = calcParams(ary)
	# update ui
	img_view = v['theFilter']
	img_view.image = uiimg
	#v['cents'].text = '%d' % nc

# apply filter
@ui.in_background
def apply_filter():
	global md
	if not hasattr(md, 'orig_ary'):
		print 'no image'
		return
	v['activity'].start()
	ary = md.orig_ary
	rgb = applyFilter(ary, md.cube, dict(mu0=md.contr))
	#rgb = nptest.applyCents(ary, md.cube, md.contr)
	uiimg = uiimg_from_array(rgb)
	img_view = v['theImage']
	img_view.image = uiimg
	v['activity'].stop()

def apply_filter_hd():
	ary = np.asarray(md.hd_img)/255.0
	rgb = applyFilter(ary, md.cube, dict(mu0=md.contr)) * 255.0
	img = Image.fromarray(rgb.astype(np.uint8))
	return img

def slider_action(sender):
	global md
	md.h_perc = v['histSlider'].value
	md.contr = v['distSlider'].value**2
	md.orig = v['dhSlider'].value
	md.h_perc = 0.2
	md.nbrs = 1
	v['minh'].text = 'foo=%.2f' % md.h_perc
	v['dmax'].text = 'contr=%.2f' % md.contr
	v['dhmin'].text = 'orig=%.2f' % md.orig
	
	# update cube
	if not hasattr(md, 'orig_ary'):
		print 'no image'
		return
	#(md.cube, nc) = nptest.calcCents(md.filt_ary)
	md.cube = calcParams(md.filt_ary)
	#v['cents'].text = '%d' % nc
	
	apply_filter()

def switch_action(sender):
	#orig = v['origSwitch'].value
	'''
	if orig:
		nptest.gain = 1.0
	else:
		nptest.gain = 0.0
	nptest.lense = v['muSwitch'].value
	'''
	slider_action(None )

@ui.in_background
def save_action(sender):
	global md
	if md.hd_img is None:
		return
	v['activity'].start()
	img = apply_filter_hd()
	photos.save_image(img)
	v['activity'].stop()
	log.warning('Image saved')

###

# init view
v = ui.load_view('test3d')

# init ActivityIndicator
ai = ui.ActivityIndicator()
ai.x = 160
ai.y = 100
ai.name = 'activity'
ai.style = ui.ACTIVITY_INDICATOR_STYLE_WHITE_LARGE
ai.hides_when_stopped = True  
v.add_subview(ai)

# init controls
#slider_action(None )
switch_action(None )

img = Image.open('orig/kueche.jpg')
open_action(None, img)
img = Image.open('orig/pond.jpg')
add_filter(None, img)

# run
v.present()

