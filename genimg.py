import Image
import ImageDraw
from colorsys import rgb_to_hsv, hsv_to_rgb

def gen_hs(v_def = 0.5, vmap = None, nc = 128):
	nh, ns = nc, nc
	width, height = 256, 256
	w,h = width / nh, height / ns
	img = Image.new('RGB', (width,height))
	drw = ImageDraw.Draw(img)
	for hue in range(nh):
		for sat in range(ns):
			x = hue * w
			y = sat * h
			if vmap:
				v_def = vmap[hue][sat]
			rgb = hsv_to_rgb(float(hue)/nh, float(sat)/ns, v_def)
			col = (int(rgb[0]*255.0), int(rgb[1]*255.0), int(rgb[2]*255.0))
			drw.rectangle([(x,y),(x+w,y+h)], fill=col)
	return img
	
def gen_3():
	width, height = 256, 256
	img = Image.new('RGB', (width,height))
	drw = ImageDraw.Draw(img)
	col = (255,0,0)
	drw.rectangle([(10,30),(50,80)], fill=col)
	col = (10,250,10)
	drw.rectangle([(100,130),(120,150)], fill=col)
	col = (15,10,240)
	drw.rectangle([(10,230),(150,280)], fill=col)

	return img
	
if __name__=='__main__':
	img = gen_hs(0.7)
	img.show()
	
