# run single analysis

import batch
import palettes
from pals import pals

pnam = 'bunt'
pal = pals[pnam]

img = palettes.load('orig/kueche.jpg')
distm = batch.distm(img, pal)

