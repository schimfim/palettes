# coding: utf-8
# todo:
#

import ui
import photos
import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import npclust3d
import logging as log

npclust3d.plot = False 

thbsize = (512,512)
hdsize = (1500,1500)

# model
class Model(object):
	pass
global md
md = Model()


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
	uiimg = uiimg_from_array(ary)
	# calc cube
	(md.cube, nc) = npclust3d.calcCube(ary, md.minh, md.dmax, md.dhmin)
	# update ui
	img_view = v['theFilter']
	img_view.image = uiimg
	v['cents'].text = '%d' % nc

# apply filter
@ui.in_background
def apply_filter():
	global md
	if not hasattr(md, 'orig_ary'):
		print 'no image'
		return
	v['activity'].start()
	ary = md.orig_ary
	rgb = npclust3d.applyCube(ary, md.cube)
	uiimg = uiimg_from_array(rgb)
	img_view = v['theImage']
	img_view.image = uiimg
	v['activity'].stop()

def slider_action(sender):
	global md
	md.minh = v['histSlider'].value ** 2.0
	md.dmax = v['distSlider'].value
	md.dhmin = v['dhSlider'].value ** 2.0
	v['minh'].text = 'minh=%.2f' % md.minh
	v['dmax'].text = 'dmax=%.2f' % md.dmax
	v['dhmin'].text = 'dhmin=%.2f' % md.dhmin
	
	# update cube
	if not hasattr(md, 'orig_ary'):
		print 'no image'
		return
	(md.cube, nc) = npclust3d.calcCube(md.orig_ary, md.minh, md.dmax, md.dhmin)
	v['cents'].text = '%d' % nc
	
	apply_filter()

def switch_action(sender):
	orig = v['origSwitch'].value
	if orig:
		npclust3d.gain = 1.0
	else:
		npclust3d.gain = 0.0
	npclust3d.lense = v['muSwitch'].value
	slider_action(None )


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

