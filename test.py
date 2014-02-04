import palettes
import Image
from pals import pals

# config
slope = 20.0 # small=large impact
palettes.xf = 0.1
palettes.defcol = (0.1,0.1,0.1)
palettes.stats = True 

print 'xf={}, slp={}'.format(palettes.xf,slope)

imgs = palettes.load_all('orig')
pnam = 'rgb'
ps = pals[pnam]
ims = imgs[2]
palettes.draw_palette({pnam:ps})
out = None 

def start():
	global out
	focus = palettes.analyse(ims, ps, gain=0.5, nperc=0.3, k=10000)
	out = palettes.adapt(ims, ps, slope, focus) 

if __name__ == '__main__':
	import cProfile
	#cProfile.run('start()')
	start()

