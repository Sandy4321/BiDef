#this script generates the input files needed for the training data during structural identification

# We're just gonna use surfaces, point defects, simple GB's and dislocations (maybe a screw with a full and partial dislocation in FCC)

# train on all of these, then show test simulations where the technique is able to identify atoms (write the code that takes any lammps file and outputs the atom types for each atom)

# make a system for adding new values to be taught

#************************************try making the strains just part of the noise in identification, without making separate identifiers******************************************

# OK so here's the training set -- 4 most popular surfaces for bcc, fcc, then a couple for hcp -- all with random strains and thermal fluctuations
# intersitial sites in each of the above structs -- all with random strains and thermal fluctuations
# simple GBs -- all with strains and random flucts
# partial and full dislocations in fcc and bcc 


import pandas as pd
from lammps import lammps
from itertools import cycle
import numpy as np
import random
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import svm

load_from_prev=True

#number of PCA components that the bi components are reduced to
num_comp=4

#how many perturbed configurations to use to simulate the effects of temperature

thermal=10

#first we make the input files for a large set of surfaces

indicies=[[[1,1,1],[-1,1,0],[-1,-1,2]], [[1,0,0],[0,1,0],[0,0,1]]]

#need teh octahedral and tetrahedral interstitial sites for bcc and fcc, need vacancies too.

interstitial=[[0.25,0.25,0.25],[0.5,0.0,0.0]]

#grain boundaries -- for a spectrum of separation angles, we'll do a separate regression to find the grain-boundary angle after identifying the struct as a GB

# RETURN TO THIS LATER

#crystal defects, we take a crystal and shear some of it by different burgers vectors, namely full dislocation a/2<110> on (111) or a/6<112>, with a stacking fault inbetween
#can just make a program that takes on NEW coefficients, defined as previously stated, with descriptors.
#the program can also auto-generate descriptors, given an initial configuration, and a number of splits to make, it will return the atoms in each split identified with PCA clustering, then ask the user for the description of those atoms, add then add them to the total list.

#we'll take the PCA and nab only the atoms which differ in the first PC as the actual defect atoms, then label them with the proper defect.
bounds=['p p s', 'p s p', 's p p']
scales=[1+x for x in np.arange(0.01,0.11,0.01)]

def get_boxes(scale):
	b=['x scale '+str(scale), ' y scale '+str(scale), ' z scale '+str(scale)]
	return [b[0],b[1],b[2], b[0]+b[1],b[0]+b[2],b[1]+b[2],b[0]+b[1]+b[2]]

def adjust_temp(lat,lx,ly,lz,xi,yi,zi,bound):

	return 'units metal\nboundary        '+str(bound)+'\nregion 		sim block -'+str(lx*5)+' '+str(lx*5)+' -'+str(ly*5)+' '+str(ly*5)+' -'+str(lz*5)+' '+str(lz*5)+'\n'+'create_box 1 sim\n'+ \
		'lattice fcc '+str(lat)+' origin 0 0 0 orient x '+str(xi[0])+' '+str(xi[1])+' '+str(xi[2])+' orient y '+ str(yi[0])+' '+str(yi[1])+' '+str(yi[2])+' orient z '+str(zi[0])+' '+str(zi[1])+' '+str(zi[2])+ '\n'+'create_atoms 1 box\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+ 'displace_atoms all random 0.2 0.2 0.2 '+str(random.randint(10,1000))+'\n'+\
		'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 8 '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 equib_8_11 id type x y z c_vb[1] c_vb[2] c_vb[3] c_vb[4] c_vb[5] c_vb[6] c_vb[7] c_vb[8] c_vb[9] c_vb[10] c_vb[11] c_vb[12] c_vb[13] c_vb[14] c_vb[15] c_vb[16] c_vb[17] c_vb[18] c_vb[19] c_vb[20] c_vb[21] c_vb[22] c_vb[23] c_vb[24] c_vb[25] c_vb[26] c_vb[27] c_vb[28] c_vb[29] c_vb[30] c_vb[31] c_vb[32] c_vb[33] c_vb[34] c_vb[35] c_vb[36] c_vb[37] c_vb[38] c_vb[39] c_vb[40] c_vb[41] c_vb[42] c_vb[43] c_vb[44] c_vb[45] c_vb[46] c_vb[47] c_vb[48] c_vb[49] c_vb[50] c_vb[51] c_vb[52] c_vb[53] c_vb[54] c_vb[55]\n'


# we need to nab the bispec components for every defect type, then merge them all into one dataframe, then standardize the coefficients, then perform the PCA

def nab_bispec_train():
	df=pd.read_csv('equib_8_11',skiprows=8,delim_whitespace=True,low_memory=False)
	# get rid of the bogus first columns
	cols=df.columns[2::]
	df.drop(df.columns[-2::],1,inplace=True)
	df.columns=cols
	return df	


# we'll separate atoms into 3 catagories, the actual surface (largest PCA deviation), the close to the surface (smaller PCA deviation), and normies (within bulk pca)
if load_from_prev==False:

	alld=pd.DataFrame()
	descriptors=[]

	for i in indicies:
		for bound in bounds:
			for s in scales:
				box=get_boxes(s)
				for b in box:
					for t in range(thermal):

						lx=np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
						ly=np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
						lz=np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)


						template=adjust_temp(3.597,lx,ly,lz,i[0],i[1],i[2],bound)
		
						with open('temp.in', 'w') as f:
							f.write(template)
							f.write('\nchange_box all '+b+' boundary '+bound+' remap units box\n')
							f.write('\nrun 0 post no\n')
				
						lmp=lammps()
						lmp.file('temp.in')
						lmp.close()
						temp=nab_bispec_train()
						alld=alld.append(temp)
						descriptors.append([i,s,b,t,bound,len(temp)])


	# we need to eliminate and consolidate identical descriptors
	# loop across and find when the surface site is the same

	def give_desc(d):
		dic={bounds[0]:0,bounds[1]:1,bounds[2]:2}
		return [d[0][dic[d[4]]],d[1],d[2]]

	d2=[]
	for d in descriptors:
		for t in range(d[5]):
			d2.append(give_desc(d))
	

	alld['desc1'] = [d[0] for d in d2]
	alld['desc2'] = [d[1] for d in d2]
	alld['desc3'] = [d[2] for d in d2]


	pickle.dump(alld, open( "alld.p", "wb" ) )

	def norm (s, m) : return lambda x : (x - m) / s

	# get pca object to save for later
	pca = PCA(n_components=num_comp)

	# now we're going to use PCA on the whole lot--and save that PCA object, then we'll consolidate the atoms, by splitting them into 3 groups (near defect, defect, and bulk)
	def consolidate_bispec(alld,pca):
		# standardize the data so the larger magnitude components don't dominate the PCA
		for x in alld.columns[5:-3]:
			alld[x] = alld[x].map(norm(alld[x].std(),alld[x].mean()))
		# get N principal components for the defects!
		lookPCA= pca.fit(alld[alld.columns[5:-3]].values)
		trans_values=pca.transform(alld[alld.columns[5:-3]].values)	
		return trans_values,pca

	newv,pca=consolidate_bispec(alld,pca)
	pickle.dump(newv, open( "newv.p", "wb"))
	pickle.dump(pca, open( "pca.p", "wb"))

else:

	alld=pickle.load(open("alld.p","rb"))
	pca=pickle.load(open("pca.p","rb"))
	newv=pickle.load(open("newv.p","rb"))

#turn the transformed bispec components into a df object

newdf=pd.DataFrame(newv)


# now we need to indentify the 3 catagories of atoms in each "set" so we group by desc
KM=KMeans(n_clusters=3)
# shift the lists to a hashable object 
alld.index=range(len(alld))
alld[alld.columns[-3]]=alld[alld.columns[-3]].astype(str)
# dict of the sorted groups
g=alld.groupby([alld.columns[-3],alld.columns[-2],alld.columns[-1]]).groups

# loop over all of the combos of strain and surface orientation
final=[]
bind={0:'surface',1:'near-surface',2:'bulk'}
for k in g:

	look=newdf.ix[g[k]]
	atom_cats=KM.fit_predict(look.values)
	
	binned=np.bincount(atom_cats).argsort()
	
	for x,c in zip(atom_cats,range(len(atom_cats))):	
		
		label=bind[binned[x]]		
		
		label=label+k[0]+' '+str(k[1])

		if k[2].find('x')!=-1:
			label=label+' x'
		elif k[2].find('y')!=-1:
			label=label+' y'
		elif k[2].find('z')!=-1:
			label=label+' z'

		final.append([g[k][c],label])	


#now we make a dictionary of classifiers

vals=list(np.unique(np.array([f[1] for f in final])))
keys=range(len(vals))

classdict=dict(zip(keys,vals))
rclassdict=dict(zip(vals,keys))

clf = svm.LinearSVC(verbose=True)

temp=newdf.values

print len(temp)

clf.fit([temp[f[0]] for f in final], [rclassdict[f[1]] for f in final])




