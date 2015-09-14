from lammps import lammps
lmp=lammps()

need_generate=False

if need_generate:

	diagonal=3
	twojmax=8
	N=865

	outer=[]
	#get all of the coefficients into outer
	for n in range(N):
		count=0
		inner=[]
		for j1 in range(0,twojmax+1):
		    if(diagonal==2):
			print j1/2,j1/2,j1/2
		    elif(diagonal==1):
			for j in range(0,min(twojmax,2*j1)+1,2):
			    print j1/2,j1/2,j/2
		    elif(diagonal==0):
			for j2 in range(0,j1+1):
			    for j in range(j1-j2,min(twojmax,j1+j2)+1,2):
				print j1/2,j2/2,j/2
				#inner.append(comp[n][count])
				count+=1
		    elif(diagonal==3):
			for j2 in range(0,j1+1):
			    for j in range(j1-j2,min(twojmax,j1+j2)+1,2):
				if (j>=j1): 
					print j1/2,j2/2,j/2
					#inner.append(comp[n][count])
					count+=1
		clen=count
		outer.append(inner)

	comm='dump myDump all custom 1 equib_8_11 id type x y z'

	for x in range(1,clen+1):
		comm=comm+' c_vb['+str(x)+']'

	with open("FCC_input.in", "a") as myfile:
		myfile.write(comm)
		myfile.write('\nrun 0')

lmp.file('FCC_input.in')

comp=lmp.extract_compute('vb',1,2)


import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
outer=np.array(outer)

# we need to normalize outer so the large magnitude values don't dominate the variance:

dft=pd.DataFrame(outer)

def norm (s, m) : return lambda x : (x - m) / s

def mod_sub (m) : return lambda x: (x - m)

good_values=[]

for x in dft.columns:
	dft[x] = dft[x].map(norm(dft[x].std(),dft[x].mean()))
	dft[x] = dft[x].map(mod_sub(dft[x].mode().values[0]))
	dft.loc[dft[x]<0.00001,x] = 0
	good_values.append(list(dft[x][dft[x]>0].index.values))

good_values=[i for s in good_values for i in s]

good_values=set(good_values)



#move back into numpy array
outer=dft.values

KM=KMeans(n_clusters=5)
pca = PCA(n_components=2)
lookPCA= pca.fit(outer)
new=pca.transform(outer)

look=KM.fit_predict(new)



from matplotlib import pyplot as plt

plt.plot([x[0] for x in new],[x[1] for x in new],'o')

#plt.show()

df=pd.read_csv('equib_8_11',skiprows=9,delim_whitespace=True,low_memory=False,names=range(6)).sort(0,ascending=True)
#df[5]=look
iden=[]
for x in range(1,len(df)+1):
	if x in good_values:
		iden.append(1)
	else:
		iden.append(0)

df[5]=iden



with open("equib_8_11") as myfile:
    head = [next(myfile) for x in xrange(9)]

with open('output', 'w') as f:
	for h in head:    
		f.write(h) 
        for x in range(len(df)):
    		for v in df[x:x+1].values[0]:
			f.write(str(v)+' ')
		f.write('\n')


#plt.show()

