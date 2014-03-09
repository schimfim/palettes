import Image
import ImageDraw
from colorsys import rgb_to_hsv, hsv_to_rgb

def gen_hs(v_def = 0.5):
	nh, ns = 32,32
	width, height = 256, 256
	w,h = width / nh, height / ns
	img = Image.new('RGB', (width,height))
	drw = ImageDraw.Draw(img)
	for hue in range(nh):
		for sat in range(ns):
			x = hue * w
			y = sat * h
			rgb = hsv_to_rgb(float(hue)/nh, float(sat)/ns, v_def)
			col = (int(rgb[0]*255.0), int(rgb[1]*255.0), int(rgb[2]*255.0))
			drw.rectangle([(x,y),(x+w,y+h)], fill=col)
	return img
	
if __name__=='__main__':
	img = gen_hs(0.7)
	img.show()
	
