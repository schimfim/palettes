from palettes import norm, load
from time import time
from math import sqrt, exp, cos, pi

normalize = True

def fdist(a, b):
	'''
	Float distance
	'''
	return sqrt(sum([(x[0]/255.0-x[1])**2 for x in zip(a,b)])/len(a))
	#return sum([(x[0]/255.0-x[1])**2 for x in zip(a,b)])
	
def distm(img, pal):
	'''
	Calc distance matrix
	'''
	idata = img.getdata()
	# normalize palette
	if normalize:
		pal = [norm(p)[0] for p in pal]
	
	print 'calc distances...'
	tic = time()
	#
	mdist = [[fdist(c, p) for p in pal] for c in idata]
	#
	toc = time()
	dt = toc-tic
	print '...done in {:.3f} secs'.format(dt)

	return mdist

sigma = 20.0
def act(dist):
    #f = exp(sigma*(1-dist))
    f = (cos((dist**2.0)*pi)/2+0.5)**20.0
    #f=1-dist**0.3
    return f
    
# softmax
def memb(distv):
	a = map(act, distv)
	den = sum(a)
	f = [b / den for b in a]
	#f = a
	return f

def membm(distv):
	'''
	Membership matrix
	'''
	print 'calc memberships...'
	tic = time()
	
	mu = [memb(d) for d in distv]
	
	toc = time()
	dt = toc-tic
	print '...done in {:.3f} secs'.format(dt)
	
	return mu

def memb2img(mu, pal):
	'''
	Calc image from membership matrix with 
	colors from palette
	'''
	print 'gen img...'
	tic = time()
	
	#cols = [[proto[i]*fc for i in [0,1,2]] for (fc, proto) in zip(mu, pal)]
	cols = [tuple([int(255.0*sum([fp*p[i] for (fp,p) in zip(f,pal)])) for i in [0,1,2]]) for f in mu]
	
	toc = time()
	dt = toc-tic
	print '...done in {:.3f} secs'.format(dt)
	
	return cols

if __name__ == '__main__':
	import Image
	pal = [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)]
	img = load('orig/kueche.jpg')
	dm = distm(img, pal)
	mu = membm(dm)
	idata = memb2img(mu, pal)
	newi = img.copy()
	newi.putdata(idata)
	
