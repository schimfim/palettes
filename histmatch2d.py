'''
Non parametric 2d-histogram matching
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

NBINS = 32
BASE = 0.0005

bin1 = np.linspace(0,1,NBINS+1)
bins = (bin1, bin1)

# Reference signal
ref_s = np.hstack((np.vstack( (np.random.normal(0.6, 0.1, 1000), np.random.normal(0.2, 0.06, 1000)) ), np.vstack( (np.random.normal(0.5, 0.1, 1800), np.random.normal(0.8, 0.1, 1800) )) ) ).T
#ref_s = (np.vstack( (np.random.normal(0.8, 0.1, 1000), np.random.normal(0.4, 0.2, 1000)) ) ).T

ref_n, foo = np.histogramdd(ref_s, bins=bins)
ref_n = ref_n / float(ref_s.shape[0]) + BASE

'''
Integrate and invert x-reference
'''
sn = np.sum(ref_n, axis=0)
sn[sn == 0] = 1
ref_intx = np.cumsum(ref_n, axis=0) / sn
# Invert
x = np.linspace(0,1,NBINS)
y = np.linspace(0,1,NBINS)
invx = np.zeros((NBINS,NBINS))
for j,yj in enumerate(y):
	for i,xi in enumerate(x):
		l = np.argwhere(xi > ref_intx[:,j])
		if len(l) == 0:
			continue 
		invx[i,j] = x[l[-1]]

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('x-Integral')
ax.set_xlabel('x')
ax.set_ylabel('y')
X,Y = np.meshgrid(x,y,indexing='ij')
ax.plot_wireframe(X,Y,ref_intx, rstride=NBINS, cstride=1)
#ax.plot_surface(X,Y,ref_intx, rstride=NBINS, cstride=1, alpha=0.5)
ax.contour(X,Y,ref_n, zdir='z', cmap='hot', offset=-0.5)
ax.set_zlim(-0.5,1)
ax.view_init(30, 120)
plt.show()

plt.clf()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('x-inverse contour')
ax.set_xlabel('x')
ax.set_ylabel('y')
X,Y = np.meshgrid(x,y,indexing='ij')
ax.plot_wireframe(X,Y,invx, rstride=NBINS, cstride=1)
ax.contour(X,Y,ref_n, zdir='z', cmap='hot', offset=-0.5)
ax.set_zlim(-0.5,1)
ax.view_init(30, 120)
plt.show()

'''
Integrate and invert y-reference
'''
sn = np.sum(ref_n, axis=1) # wg axis=1
sn[sn == 0] = 1
ref_inty = (np.cumsum(ref_n, axis=1).T / sn).T # wg axis=1!
# Invert
x = np.linspace(0,1,NBINS)
y = np.linspace(0,1,NBINS)
invy = np.zeros((NBINS,NBINS))
for i,xi in enumerate(x):
	for j,yj in enumerate(y):
		l = np.argwhere(yj > ref_inty[i,:])
		if len(l) == 0:
			continue 
		invy[i,j] = y[l[-1]]

plt.clf()
plt.title('y-Integral')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
X,Y = np.meshgrid(x,y,indexing='ij')
ax.plot_wireframe(X,Y,ref_inty, rstride=1, cstride=NBINS)
ax.contour(X,Y,ref_n, zdir='z', cmap='hot', offset=-0.5)
ax.set_zlim(-0.5,1)
ax.view_init(30, -120)
plt.show()

plt.clf()
plt.title('x-inverse contour')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
X,Y = np.meshgrid(x,y,indexing='ij')
ax.plot_wireframe(X,Y,invy, rstride=1, cstride=NBINS)
ax.contour(X,Y,ref_n, zdir='z', cmap='hot', offset=-0.5)
ax.set_zlim(-0.5,1)
ax.view_init(30, -120)
plt.show()

'''
Process input
'''
# Input signal
in_s = np.random.random(size=(2000,2))

# Process
idx=np.asarray(in_s*(NBINS),dtype=np.integer)
out_s0 = invx[idx[:,0],idx[:,1]]
out_s1 = invy[idx[:,0],idx[:,1]]
out_s = np.vstack((out_s0,out_s1)).T
out_n, foo = np.histogramdd(out_s, bins=bins)
out_n = out_n / float(out_s.shape[0])

plt.clf()
plt.title('output distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.contour(X,Y,out_n, cmap='autumn', linewidths=2)
plt.contour(X,Y,ref_n, cmap='winter')
plt.show()

# stream
x0 = np.linspace(0,1,16)
xx,yy = np.meshgrid(x0,x0,indexing='ij')
xidx=np.asarray(xx*(NBINS-1),dtype=np.integer)
yidx=np.asarray(yy*(NBINS-1),dtype=np.integer)
strx = invx[xidx,yidx] - xx
stry = invy[xidx,yidx] - yy

plt.clf()
plt.title('streams')
plt.xlabel('x')
plt.ylabel('y')
plt.quiver(xx,yy,strx,stry)
plt.show()
