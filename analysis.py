'''
Small helper script to analyze bi-sepctrum components in a single system. For only one file.
'''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
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

df=pd.read_csv('equib_8_11',skiprows=8,delim_whitespace=True,low_memory=False)
# get rid of the bogus first columns
cols=df.columns[2::]
df.drop(df.columns[-2::],1,inplace=True)
df.columns=cols
# get the principal components for the defects!
def norm (s, m) : return lambda x : (x - m) / s
for x in df.columns[5::]:
	df[x] = df[x].map(norm(df[x].std(),df[x].mean()))

num_comp=2

pca = PCA(n_components=num_comp)
lookPCA= pca.fit(df[df.columns[5::]].values)
trans_values=pca.transform(df[df.columns[5::]].values)

with open("equib_8_11") as myfile:
    head = [next(myfile) for x in xrange(8)]

with open('output', 'w') as f:
	for h in head:    
		f.write(h) 
	f.write('ITEM: ATOMS id type x y z ') 
	for x in range(num_comp):
		f.write('v'+str(x)+' ')
	f.write('\n')
        for x in range(len(df)):
		for v in df[x:x+1].values[0][:5]:
			f.write(str(v)+' ')
    		for v in trans_values[x]:
			f.write(str(v)+' ')
		f.write('\n')




