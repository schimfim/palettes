import clipboard
import re

t = clipboard.get()
items = re.findall('rgb\((.+),(.+),(.+)\)', t)
vals = [(float(i[0])/255.0, float(i[1])/255.0, float(i[2])/255.0) for i in items]
#rgb = '[' + ','.join(vals) + ']'
rgb = repr(vals)
clipboard.set(rgb)
print 'set clipboard to:', rgb

'''
Color Name	Hex	RGB
Cherry Berry	#d8151b	rgb(216, 21, 27)
Orange Chorus	#f76114	rgb(247, 97, 20)
gothumai	#dcaa25	rgb(220, 170, 37)
grassy	#129d28	rgb(18, 157, 40)
Blu Missoni	#2459a7	rgb(36, 89, 167)

==>

[('0.847058823529', '0.0823529411765', '0.105882352941'), ('0.96862745098', '0.380392156863', '0.078431372549'), ('0.862745098039', '0.666666666667', '0.145098039216'), ('0.0705882352941', '0.61568627451', '0.156862745098'), ('0.141176470588', '0.349019607843', '0.654901960784')]
'''

