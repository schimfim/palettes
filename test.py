import palettes
import Image
from pals import pals

# config
slope = 20.0 # small=large impact
palettes.xf = 1.0
palettes.defcol = (0.0,0.0,0.0)
palettes.stats = True 
size = (256,256)

print 'xf={}, slp={}'.format(palettes.xf,slope)

imgs = palettes.load_all('orig',size)
pnam = 'bunt'
ps = pals[pnam]
ims = imgs[1]
palettes.draw_palette({pnam:ps})
out = None 

def start():
	global out
	focus = palettes.analyse(ims, ps, gain=0.7, nperc=0.01, k=10000)
	out = palettes.adapt(ims, ps, slope, focus) 

if __name__ == '__main__':
	import cProfile
	#cProfile.run('start()')
	start()

