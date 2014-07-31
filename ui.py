# coding: utf-8
# todo:
# take photo
# show original
# activity indicator (x)
# select algorithm
# save to camera roll (x)
# save/load state (x)
# auto gain (how?)

import ui
import photos
import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import np1d
import logging as log

thbsize = (384,384)
hdsize = (1500,1500)

# model
global md
md = dict(orig_img=None,
                    orig_ary=None,
                    filt_img=None,
                    filt_ary=None,
                    bal=0.5
                    )

def uiimg_from_array(ary):
	# ary in (0,1)
	int_ary = ary*255
	int_ary = int_ary.astype(np.uint8)
	plt.imsave('tmp.png', int_ary, format='png')
	uimg = ui.Image.named('tmp.png')
	return uimg

# open image
@ui.in_background
def open_action(sender):
	global md
	img = photos.pick_image()
	md['hd_img'] = img.copy()
	md['hd_img'].thumbnail(hdsize)
	print img.size
	img.thumbnail(thbsize)
	print md['hd_img'].size
	md['orig_ary'] = np.asarray(img)/255.0
	md['orig_img'] = uiimg_from_array(md['orig_ary'])
	# update ui
	img_view = v['theImage']
	img_view.image = md['orig_img']

def pick_image():
	img = photos.pick_image()
	img.thumbnail((256,256))
	return img

# add filter
@ui.in_background
def add_filter(sender):
	img = pick_image()
	ary = np.asarray(img)/255.0
	uiimg = uiimg_from_array(ary)
	
	hsv = rgb_to_hsv(ary)
	maps = np1d.gen_maps(hsv)
	#plt.show()
	
	cells = v['tableview1'].data_source.items
	cells.append({'title':'Hallo!', 'image':uiimg, 'maps':maps, 'ary':ary})
	save_filters()
	
# apply filter
@ui.in_background
def apply_filter():
	global md
	if md['orig_ary'] is None:
		return
	v['activity'].start()
	hsv = rgb_to_hsv(md['orig_ary'])
	hsv_filt = np1d.apply_maps(hsv, md['maps'], md['bal'])
	rgb = hsv_to_rgb(hsv_filt)
	uiimg = uiimg_from_array(rgb)
	
	md['filt_ary'] = rgb
	md['filt_img'] = uiimg
	img_view = v['theImage']
	img_view.image = uiimg
	v['activity'].stop()

# apply filter to hd img
# returns PIL image
def apply_filter_hd():
	global md
	ary = np.asarray(md['hd_img'])/255.0
	print ary.shape
	hsv = rgb_to_hsv(ary)
	hsv_filt = np1d.apply_maps(hsv, md['maps'], md['bal'])
	rgb = hsv_to_rgb(hsv_filt) * 255
	# test rgb = hsv_to_rgb(hsv) * 255
	img = Image.fromarray(rgb.astype(np.uint8, copy=False))
	return img

# select filter
def filter_action(sender):
	global md
	cell = sender.items[sender.selected_row]
	md['maps'] = cell['maps']
	apply_filter()

def slider_action(sender):
	global md
	md['bal'] = v['slider1'].value
	apply_filter()

@ui.in_background
def save_action(sender):
	global md
	if md['hd_img'] is None:
		return
	v['activity'].start()
	img = apply_filter_hd()
	photos.save_image(img)
	v['activity'].stop()
	log.warning('Image saved')

def edit_action(sender):
	save_filters()

def save_filters():
	cells = v['tableview1'].data_source.items
	if len(cells) == 0:
		return 
	maps = []
	arys = []
	for i in range(len(cells)):
		maps.append(cells[i]['maps'])
		arys.append(cells[i]['ary'])
	np.savez('maps.npz', maps=maps)
	np.savez('arys.npz', arys=arys)

def load_filters(cells):
	# load maps
	try: 
		npz = np.load('maps.npz')
	except IOError:
		return 
	maps = npz['maps']
	npz.close()
	print len(maps)
	# load arrays
	npz = np.load('arys.npz')
	arys = npz['arys']
	npz.close()
	# create cells
	for [map,ary] in zip(maps,arys):
		uiimg = uiimg_from_array(ary)
		cells.append({'title':'Hallo!', 'image':uiimg, 'maps':map, 'ary':ary}) 

###

# init view
v = ui.load_view('ui')

# init ActivityIndicator
ai = ui.ActivityIndicator()
ai.x = 160
ai.y = 100
ai.name = 'activity'
ai.style = ui.ACTIVITY_INDICATOR_STYLE_WHITE_LARGE
ai.hides_when_stopped = True  
v.add_subview(ai)

# init filter table
cells = v['tableview1'].data_source.items
del cells[0] # auto created by ui editor
load_filters(cells)

# init slider
slider_action(v['slider1'])

# run
v.present()

