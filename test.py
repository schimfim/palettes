from pprint import pprint

mu = 0.5

def cum_dist(freq):
	cd = []
	sumf = 0
	for x in freq:
		sumf += x
		cd.append(sumf)
	return cd

def gen_row(freq):
	cd = cum_dist(freq)
	sf = sum(cd)
	row = [x/sf for x in cd]
	return row

def field(sats, freq):
	n = len(sats)
	fld = [0.0] * n
	new_fq = [0.0] * n
	sats.insert(0,0.0)
	sats.append(0.0)
	freq.insert(0,0.0)
	freq.append(0.0)
	for i in range(1,n+1):
		fld[i-1] = (mu*sats[i-1]*freq[i-1] + sats[i]*freq[i] + mu*sats[i+1]*freq[i+1]) / (mu*freq[i-1] + freq[i] + mu*freq[i+1])
		new_fq[i-1] = (mu*freq[i-1] + freq[i] + mu*freq[i+1]) / (1+2*mu)
	
	return fld, new_fq



if __name__ == '__main__':
	sats = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
	freq = [0.2, 0.3, 0.9, 0.4, 0.6, 0.2]
	#freq = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	
	r = gen_row(freq)
	cd = [round(x,2) for x in r]
	print cd
