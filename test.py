import palettes
import Image
from pals import pals

# algo params
slope = 20.0 # small=large impact
palettes.xf = 1.0
palettes.defcol = (0.0,0.0,0.0)
gain=0.5
nperc=0.05
palettes.normalize = False          
param_str = 'np={:.2f} gn={:.2f} sl={:.1f} nm={}'.format(nperc, gain, slope, palettes.normalize)
param_str2 = 'xf={:.2f} dc={}'.format(palettes.xf, palettes.defcol)

# config
palettes.stats = False  
size = (256,256)

print param_str

imgs = palettes.load_all('orig',size)
pnam = 'rgb'
ps = pals[pnam]
ims = imgs[2]
pal_img = palettes.draw_palette({pnam:ps})
out = None 

def start():
	global out
	focus = palettes.analyse(ims, ps, gain=0.5, nperc=0.05, k=10000)
	out = palettes.adapt(ims, ps, slope, focus) 
	palettes.add_palette(out, pal_img)
	palettes.add_caption(out, param_str, param_str2)

if __name__ == '__main__':
	import cProfile
	#cProfile.run('start()')
	start()

