'''
Non parametric histogram matching
'''

import numpy as np
import matplotlib.pyplot as plt

NBINS = 32
bins = np.linspace(0,1,NBINS)

# Reference signal
ref_s = np.concatenate(( np.random.normal(0.6, 0.1, 1000), np.random.normal(0.2, 0.06, 500) ))

ref_n, foo = np.histogram(ref_s, bins=bins)
ref_n = ref_n / float(len(ref_s))

# Input signal
in_s = np.random.random(size=2000)
in_n, foo = np.histogram(in_s, bins=bins)
in_n = in_n / float(len(in_s))

# Integrate reference
ref_int = np.cumsum(ref_n)
# Invert
x = np.linspace(0,1,len(ref_int))
inv = np.zeros_like(x)
for i,xi in enumerate(x):
	l = np.argwhere(xi > ref_int)
	if len(l) == 0:
		continue 
	inv[i] = x[l[-1]]

# Process input signal
idx=np.asarray(in_s*(len(inv)-1),dtype=np.integer)
out_s = inv[idx]
out_n, foo = np.histogram(out_s, bins=bins)
out_n = out_n / float(len(out_s))

plt.clf()
plt.title('Integral and inverse')
plt.axis([0,1,0,1.1])
plt.bar(bins[:-1], ref_int, width=1.0/NBINS)
plt.bar(bins[:-1], inv, width=1.0/NBINS, color='red', alpha=0.5)
plt.show()

plt.clf()
plt.title('Input, reference and output distribution')
plt.axis([0,1,0,max(out_n)])
plt.bar(bins[:-1], in_n, width=1.0/NBINS, color='blue', alpha=1.0)
plt.bar(bins[:-1], ref_n, width=1.0/NBINS, color='green', alpha=1.0)
plt.bar(bins[:-1], out_n, width=1.0/NBINS, color='red', alpha=0.5)
plt.legend(('in','ref','out'))
plt.show()

