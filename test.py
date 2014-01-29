import palettes
import Image
from pals import pals

# config
slope = 8.0 # small=large impact
focus = 0.3 # small=narrow range
palettes.xf = 1.0
palettes.sigma = 10.0
print 'xf={}, slp={}, fcs={}, sig={}'.format(palettes.xf,slope,focus,palettes.sigma)

blu = [(0.0,0.0,1.0)]
#red = [(1.0,0.0,0.0)]
rgb = [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)]

imgs = palettes.load_all('orig')
#npals = [palettes.calc_palette(pal_img, 5) for pal_img in imgs]
ps = pals[0]
ps = blu
ims = imgs[1]
palettes.draw_palette(ps)

out = [palettes.adapt(img, pal, slope, focus) for img in [ims] for pal in [ps]]
#palettes.save_all(out,'out')

