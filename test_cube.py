from pprint import pprint
from hues import Filter, rot
from colorsys import rgb_to_hsv, hsv_to_rgb

filt = Filter(8)
filt.hues = [0.01, 0.135, 0.26, 0.385, 0.51, 0.635, 0.76, 0.835]
filt.update()
print 'match=', filt.match
print 'focus=', filt.focus

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
			(rn,gn,bn) = hsv_to_rgb(h,s,v)
			print("%.4f %.4f %.4f - %.4f %.4f %.4f" % (r,g,b,rn,gn,bn))
			##
			v_z.append((rn,gn,bn))
		v_y.append(v_z)
	v_x.append(v_y)

# transform

