#this script generates the input files needed for the training data during structural identification

# We're just gonna use surfaces, point defects, simple GB's and dislocations (maybe a screw with a full and partial dislocation in FCC)

# train on all of these, then show test simulations where the technique is able to identify atoms (write the code that takes any lammps file and outputs the atom types for each atom)

# make a system for adding new values to be taught

#************************************try making the strains just part of the noise in identification, without making separate identifiers******************************************

# OK so here's the training set -- 4 most popular surfaces for bcc, fcc, then a couple for hcp -- all with random strains and thermal fluctuations
# intersitial sites in each of the above structs -- all with random strains and thermal fluctuations
# simple GBs -- all with strains and random flucts
# partial and full dislocations in fcc and bcc 



# 9/19: Prototype now works

# ToDo:
# regenerate dislocation structures -- make them the correct size so we don't have to rescale them.
# Add in gain boundaries -- they should be rather simple, we'll generate one large cell and use iterations of that for the GB
# write the UX code which returns each atom with a descriptor.
# try introducing more temerature variations in the samples, now that we've verified it works, this way we can have an
#advantage over other methods!!
# Use method on some live strain tests which I have -- should be able to identify surface atoms, ect.


#then for writing -- make the introduction include the scope: reduces time intesive analysis, allows for cells to large to investigate manually, and finally gives the ability for simulations
#to section off parts with a particular behavior to use a different set of potentials or a more advanced method.

#Also make the algorithmic description, using the PCA to illustrate the point as the simulation progresses. 



import pandas as pd
from itertools import cycle
import numpy as np
import random
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import preprocessing
from feature_structures import *
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
from sklearn.externals import joblib

load_from_prev=False
need_seperate_clfs=False

#grain boundaries -- for a spectrum of separation angles, we'll do a separate regression to find the grain-boundary angle after identifying the struct as a GB

# RETURN TO THIS LATER

#crystal defects, we take a crystal and shear some of it by different burgers vectors, namely full dislocation a/2<110> on (111) or a/6<112>, with a stacking fault inbetween

def norm (s, m) : return lambda x : (x - m) / s

class simulation_env:
     def __init__(self):
	 self.sampling_rate = 0.001#how often we select for keeps
	 self.num_comp = 5#number of PCA components that the bi components are reduced to
	 self.sample_size=100000#size of samples when machine learning
	 self.thermal=10 #number of thermal perturbations for each config.
	 self.lattice=3.597 #lattice size
	 self.indicies=[[[1,1,1],[-1,1,0],[-1,-1,2]], [[0,1,0],[0,0,1],[1,0,0]]]#crystal orientations
	 self.interstitial={'fcc':[[0.25,0.25,0.25],[0.5,0.0,0.0]],'bcc':[[0.5,0.25,0.0],[0.5,0.5,0.0]]}# interstitial sites for fcc, bcc (tet then oct)
	 self.structs=['fcc','bcc'] #structures considered
	 self.bounds=['p p s', 'p s p', 's p p'] #boundaries
	 self.scales=[1+x for x in np.arange(0.01,0.11,0.05)] #scales for straining the simulation cell
	 self.defect_names=['fccfull.lmp','fccpartial.lmp','bcc_disloc.lmp']
	 self.flocs={self.defect_names[0]:'fccfull_temp.structures',self.defect_names[2]:'BCC_temp.structures',self.defect_names[1]:'temp.structures'} #dislocation prototype file structural analysis files
	 self.grain_names=['fcc_polycrystal.lmp','bcc_polycrystal.lmp']
	
#f has the each of the files to consider, tdict relates the file name to the defect name
	#condit contains the conditions for each simulation, surf or no surf, CNA or CSP [0] spot
	# is the description and the [1] spot is the value needed

se=simulation_env()


# Use PCA to compress the representation of the bispectrum
def consolidate_bispec(alld,pca,rclassdict):
	# standardize the data so the larger magnitude components don't dominate the PCA
	# scaler=preprocessing.StandardScaler().fit()
	for x in alld.columns[5:-1]:
		alld[x] = alld[x].map(norm(alld[x].std(),alld[x].mean()))
	# get N principal components for the defects!
	lookPCA= pca.fit(alld[alld.columns[5:-1]].values)
	trans_values=pca.transform(alld[alld.columns[5:-1]].values)	
	trans_values=[np.append(t,rclassdict[alld['desc'].values[x]]) for x,t in zip(range(len(alld)),trans_values)]
	return trans_values,pca

#pca free version, aka all bispectrum coefficients are used
def consolidate_data(alld,pca,rclassdict):
	# standardize the data so the larger magnitude components don't dominate the PCA
	# scaler=preprocessing.StandardScaler().fit()
	for x in alld.columns[5:-1]:
		alld[x] = alld[x].map(norm(alld[x].std(),alld[x].mean()))
	# get N principal components for the defects!
	#lookPCA= pca.fit(alld[alld.columns[5:-1]].values)
	#trans_values=pca.transform(alld[alld.columns[5:-1]].values)	
	trans_values=alld[alld.columns[5:-1]].values
	trans_values=[np.append(t,rclassdict[alld['desc'].values[x]]) for x,t in zip(range(len(alld)),trans_values)]
	return trans_values,pca



def dump_stats(alld):
	for x in alld.columns[5:-1]:
		m=alld[x].mean()
		s=alld[x].std()
		pickle.dump(m, open( "./data_stats/"+str(x)+"_m.p", "wb"))
		pickle.dump(s, open( "./data_stats/"+str(x)+"_s.p", "wb"))
	pass

# pull away the surface atoms to look at an interior defect when only periodic in z
def remove_surface_atoms (df):
	ind=df[df['x'].max()-df['x']>5].index.values
	ind=np.intersect1d(ind,df[df['x']-df['x'].min()>5].index.values)
	ind=np.intersect1d(ind,df[df['y'].max()-df['y']>5].index.values)
	ind=np.intersect1d(ind,df[df['y']-df['y'].min()>5].index.values)
	#ind=np.intersect1d(ind,df[df['z'].max()-df['z']>5].index.values)
	#ind=np.intersect1d(ind,df[df['z']-df['z'].min()>5].index.values)
	return ind

def remove_surface_atoms_wZ (df):
	ind=df[df['x'].max()-df['x']>5].index.values
	ind=np.intersect1d(ind,df[df['x']-df['x'].min()>5].index.values)
	ind=np.intersect1d(ind,df[df['y'].max()-df['y']>5].index.values)
	ind=np.intersect1d(ind,df[df['y']-df['y'].min()>5].index.values)
	ind=np.intersect1d(ind,df[df['z'].max()-df['z']>3].index.values)
	ind=np.intersect1d(ind,df[df['z']-df['z'].min()>3].index.values)
	return ind

#All we need to give are iterators and post boxes

if load_from_prev==False:
	
	# run the data through the pipeline
	alld,descriptors,DOUT=make_all_structures(se)
	pickle.dump(alld, open( "alld.p", "wb" ) )
	# get pca object to save for later
	pca = PCA(n_components=se.num_comp)
	# dump the stats for each column, so they can be re-mapped in future simulations
	dump_stats(alld)

	vals=list(np.unique(alld['desc'].values))
	keys=range(len(vals))
	classdict=dict(zip(keys,vals))
	rclassdict=dict(zip(vals,keys))
	# no more PCA is performed.
	newv,pca=consolidate_data(alld,pca,rclassdict)
	fdf=pd.DataFrame(newv)
	pickle.dump(fdf, open( "fdf.p", "wb"))
	pickle.dump(newv, open( "newv.p", "wb"))
	pickle.dump(pca, open( "pca.p", "wb"))

else:
	fdf=pickle.load(open("fdf.p","rb"))
	alld=pickle.load(open("alld.p","rb"))
	pca=pickle.load(open("pca.p","rb"))
	newv=pickle.load(open("newv.p","rb"))

#turn the transformed bispec components into a df object

newdf=pd.DataFrame(newv)

#now we make a dictionary of classifiers

vals=list(np.unique(alld['desc'].values))
keys=range(len(vals))

print '\n There are '+str(len(vals))+' options.'

classdict=dict(zip(keys,vals))
rclassdict=dict(zip(vals,keys))


pickle.dump(classdict, open( "classdict.p", "wb"))

from sklearn.ensemble import RandomForestClassifier


def train_ML(clf,X,Y,vals=vals,trim=False):
	if trim==True:
		#get an even representation from each possible output
		amt=min([np.where(np.array(Y)==x)[0].shape[0] for x in range(len(vals))])
		from random import shuffle
		Xt=[]
		Yt=[]
		for r in range(len(vals)):
			indtemp=np.where(np.array(Y)==r)[0]
			if len(indtemp) < se.sample_size:
				index_shuf=range(len(indtemp))
			else:
				index_shuf=range(se.sample_size)
			shuffle(index_shuf)
			Xt.append([X[indtemp[i]] for i in index_shuf])
			Yt.append([Y[indtemp[i]] for i in index_shuf])
		Xs=[item for sublist in Xt for item in sublist]
		Ys=[item for sublist in Yt for item in sublist]
	else:
		Xs=X
		Ys=Y
	clf.fit(Xs,Ys)
	print clf.score(Xs,Ys)
	return clf

#assign final feature and descriptors

if need_seperate_clfs==True:
	clf_list=[RandomForestClassifier(n_estimators=1000, min_samples_leaf=10, verbose=True) for x in range(len(classdict.keys()))]
	clfdict={}
	for k in classdict.keys():
		print "\n Training ..."
		if classdict[k].find('bulk')==-1:
			if classdict[k].find('fcc')!=-1:
				comp=9
			elif classdict[k].find('bcc')!=-1:
				comp=8
			Y=fdf[(fdf[55]==k)|(fdf[55]==comp)[::100]][55].values
			X=fdf[(fdf[55]==k)|(fdf[55]==comp)[::100]][fdf.columns[:55]].values
		clfdict[k]=train_ML(clf_list[k],X,Y)	
	#now save em
	for k in classdict.keys():
		joblib.dump(clfdict[k], './pickled/'+classdict[k]+'.pkl') 
else:
	clftot=RandomForestClassifier(n_estimators=1000, min_samples_leaf=10, verbose=True)
	X=fdf[fdf.columns[:55]].values[::100]
	Y=fdf[fdf.columns[55]].values[::100]
	clftot.fit(X,Y)
	joblib.dump(clftot, './pickled/clftot.pkl')

#allow the trees to be assigned according to each binary defect
#X=newdf[newdf.columns[:se.num_comp]].values
#Y=newdf[se.num_comp].values




#for if pdf is made with only 2 PCA's



#joblib.dump(clf, './pickled/rf.pkl') 


def plot_def(xlo,xhi,ylo,yhi):
		plt.cla()
		plt.clf()
		color=cm.rainbow(np.linspace(0,1.0,5))
		markers = ['.','*','s','d','p']
		params=list(itertools.product(*[color,markers]))
		for ck,m in zip(classdict.keys(),params[:len(classdict.keys())]):
			if classdict[ck].find('bulk')!=-1:
				if classdict[ck].find('fcc')!=-1:
					lilc=color[0]
				else:
					lilc=color[4]
				plt.scatter(pdf[pdf[2]==ck][0].values[::90],pdf[pdf[2]==ck][1].values[::90],marker='o',c=lilc,s=130,label=classdict[ck])
			#c=next(color)
			#else:
				#plt.scatter(pdf[pdf[2]==ck][0].values[:1000],pdf[pdf[2]==ck][1].values[:1000],marker=m[1],c=m[0],s=100,label=classdict[ck])
		plt.axis([xlo,xhi,ylo,yhi], fontsize=24)	
		plt.legend(bbox_to_anchor=(1.2,1.0),prop={'size':16})
		plt.show()


