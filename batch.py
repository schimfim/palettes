import palettes
import Image
from pals import pals

# config
slope = 20.0 # small=large impact
focus = 1.0 # small=narrow range
palettes.xf = 1.0
palettes.sigma = 10.0
palettes.stats = True  
print 'xf={}, slp={}, fcs={}, sig={}'.format(palettes.xf,slope,focus,palettes.sigma)

blu = [(0.0,0.0,1.0)]
red = [(1.0,0.0,0.0)]
rgb = [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)]

imgs = palettes.load_all('orig')
npals = [palettes.calc_palette(pal_img, 5) for pal_img in imgs]
ps = pals[0]
ps = rgb
ims = imgs[2]
palettes.draw_palette(*npals)

out = [palettes.adapt(img, pal, slope, focus) for img in imgs for pal in npals]
#palettes.save_all(out,'out')

