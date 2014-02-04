import palettes
import Image
from pals import pals

# config
slope = 20.0 # small=large impact
focus = 1.0 # small=narrow range
palettes.xf = 1.0
palettes.sigma = 10.0  
print 'xf={}, slp={}, fcs={}, sig={}'.format(palettes.xf,slope,focus,palettes.sigma)

blu = [(0.0,0.0,1.0)]
red = [(1.0,0.0,0.0)]
rgb = [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)]

imgs = palettes.load_all('orig')
npals = [palettes.calc_palette(pal_img, 5) for pal_img in imgs]
palettes.draw_palette(*npals)

out = []
for img in imgs:
	for pal in npals:
		focus = palettes.analyse(img, pal, 10000)
		out.append(palettes.adapt(img, pal, slope, focus))
	
palettes.save_all(out,'out')

