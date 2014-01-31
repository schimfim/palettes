import palettes
import Image
from pals import pals

# config
slope = 20.0 # small=large impact
focus = 1.3 # small=narrow range
palettes.xf = 1.0
palettes.sigma = 10.0
palettes.stats = True  
print 'xf={}, slp={}, fcs={}, sig={}'.format(palettes.xf,slope,focus,palettes.sigma)

blu = [(0.0,0.0,1.0)]
red = [(1.0,0.0,0.0)]
rgb = [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)]

imgs = palettes.load_all('orig')
ps = pals[0]
ims = imgs[2]
palettes.draw_palette(ps)
out = None 

def start():
	global out
	out = palettes.adapt(ims, ps, slope, focus) 

if __name__ == '__main__':
	import cProfile
	#cProfile.run('start()')
	start()

