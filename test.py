import palettes
import Image
from pals import pals
import time

# algo params
slope = 20.0 # small=large impact
palettes.xf = 1.0
palettes.defcol = (0.0,0.0,0.0)
gain=0.5
nperc=0.05
palettes.normalize = True           
param_str = 'np={:.2f} gn={:.2f} sl={:.1f} nm={}'.format(nperc, gain, slope, palettes.normalize)
param_str2 = 'xf={:.2f} dc={}'.format(palettes.xf, palettes.defcol)

# config
palettes.stats = False  
size = (128,128)

print param_str

imgs = palettes.load_all('orig',size)
#imgs = [imgs[1]]
pnam = 'blu'
ps = pals[pnam]
pal_img = palettes.draw_palette({pnam:ps})
out = None 

start_time = time.strftime('%Y-%m-%d_%H:%M')
fname_base = 'img_{}_{}_'.format(start_time, pnam)

def start():
	global out #?
	out_imgs = []
	for i,ims in enumerate(imgs):
		focus = palettes.analyse(ims, ps, gain=0.5, nperc=0.05, k=10000)
		out = palettes.adapt(ims, ps, slope, focus) 
		palettes.add_palette(out, pal_img)
		palettes.add_caption(out, param_str, param_str2)
		out_imgs.append(out)
		
	palettes.save_all(out_imgs,'out', fname_base)

if __name__ == '__main__':
	import cProfile
	#cProfile.run('start()')
	start()

