from pprint import pprint
from hues import Filter, rot
from colorsys import rgb_to_hsv, hsv_to_rgb

filt = Filter(4)
filt.hues = [0.0,0.1,0.2,0.3]
filt.update()

order = 4
# init cube
v_x = []
for x in range(order):
	b = float(x)/(order-1)
	v_y = []
	for y in range(order):
		g = float(y)/(order-1)
		v_z = []
		for z in range(order):
			r = float(z)/(order-1)
			## xform
			(h,s,v) = rgb_to_hsv(r,g,b)
			(h,s,v) = rot((h,s,v), filt)
			(r,g,b) = hsv_to_rgb(h,s,v)
			##
			v_z.append((r,g,b))
		v_y.append(v_z)
	v_x.append(v_y)

pprint(v_x)

# transform

