import palettes
import Image
from pals import pals
import time

# algo params
slope = 20.0 # small=large impact
palettes.xf = 1.0
palettes.defcol = (0.0,0.0,0.0)
palettes.normalize = True

# config
palettes.stats = False
palettes.thumbnail = True
size = (256,256)

imgs = palettes.load_all('orig',size)
imgs = [imgs[1]]
pnam = 'bunt'
ps = pals[pnam]
pal_img = palettes.draw_palette({pnam:ps})
out = None 

start_time = time.strftime('%Y-%m-%d_%H:%M')
prefix = ''
folder = 'out'
fname_base = prefix + 'img_{}_{}_'.format(start_time, pnam)

def start():
	global out #?
	out_imgs = []
	for nperc in [0.02]:
		for gain in [0.2]:
			for i,ims in enumerate(imgs):
				focus = palettes.analyse(ims, ps, gain=gain, nperc=nperc)
				out = palettes.adapt(ims, ps, slope, focus)

				palettes.add_palette(out, pal_img)
				param_str = 'np={:.2f} gn={:.2f} sl={:.1f} nm={}'.format(nperc, gain, slope, palettes.normalize)
				param_str2 = 'xf={:.2f} dc={}'.format(palettes.xf, palettes.defcol)
				print param_str

				palettes.add_caption(out, param_str, param_str2)
				out_imgs.append(out)
		
	palettes.save_all(out_imgs,folder, fname_base)

if __name__ == '__main__':
	import cProfile
	#cProfile.run('start()')
	start()

