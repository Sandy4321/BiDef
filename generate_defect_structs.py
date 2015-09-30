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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from random import shuffle
from matplotlib.ticker import MaxNLocator
mpl.rcdefaults()
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.linewidth'] = 2 # set the value globally
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1

load_from_prev=False
need_seperate_clfs=False
clf_num=1

#grain boundaries -- for a spectrum of separation angles, we'll do a separate regression to find the grain-boundary angle after identifying the struct as a GB

# RETURN TO THIS LATER

#crystal defects, we take a crystal and shear some of it by different burgers vectors, namely full dislocation a/2<110> on (111) or a/6<112>, with a stacking fault inbetween

def norm (s, m) : return lambda x : (x - m) / s

class simulation_env:
     def __init__(self):
	 self.sampling_rate = 0.0000001#how often we select for keeps
	 self.num_comp = 5#number of PCA components that the bi components are reduced to
	 self.sample_size=10000#size of samples when machine learning
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
def consolidate_bispec(fdf,pca,rclassdict):
	# standardize the data so the larger magnitude components don't dominate the PCA
	# scaler=preprocessing.StandardScaler().fit()
	for x in fdf.columns[5:-1]:
		fdf[x] = fdf[x].map(norm(fdf[x].std(),fdf[x].mean()))
	# get N principal components for the defects!
	lookPCA= pca.fit(fdf[fdf.columns[5:-1]].values)
	trans_values=pca.transform(fdf[fdf.columns[5:-1]].values)	
	trans_values=[np.append(t,fdf[bispec_dict[8]].values[x]) for x,t in zip(range(len(fdf)),trans_values)]
	
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


from sklearn.ensemble import RandomForestClassifier


def train_ML(clf,X,Y,trim=False,get_test_set=False,need_fit=True):
	
	train_set_size=se.sample_size

	if trim==True:
		#get an even representation from each possible output
		amt=min([np.where(np.array(Y)==x)[0].shape[0] for x in range(len(np.unique(np.array(Y))))])
		print str(amt) + ' max number of samples'
		if get_test_set: train_set_size=int(amt*0.75)

		Xt=[]
		Yt=[]
		Xv=[]
		Yv=[]

		for r in range(len(np.unique(np.array(Y)))):
			indtemp=np.where(np.array(Y)==r)
			if len(indtemp[0]) < train_set_size:
				maxi=len(indtemp[0])
			else:
				maxi=train_set_size
			s=range(len(indtemp[0]))
			shuffle(s)
			Xt.append([X[indtemp[0][i]] for i in s[:maxi]])
			Yt.append([Y[indtemp[0][i]] for i in s[:maxi]])
			if get_test_set:
				Xv.append([X[indtemp[0][i]] for i in s[maxi:se.sample_size]])
				Yv.append([Y[indtemp[0][i]] for i in s[maxi:se.sample_size]])
		Xs=[item for sublist in Xt for item in sublist]
		Ys=[item for sublist in Yt for item in sublist]
		if get_test_set:
			Xv=[item for sublist in Xv for item in sublist]
			Yv=[item for sublist in Yv for item in sublist]
	else:
		Xs=X
		Ys=Y
	if need_fit: clf.fit(Xs,Ys)
	sc=clf.score(Xs,Ys)
	print sc
	if get_test_set:	
		AS=accuracy_score(Yv,clf.predict(Xv))
		print AS
	else:
		AS=[]
	return clf,Xs,Ys,sc,Xv,Yv,AS

def plot_def(pdf,xlo,xhi,ylo,yhi,classdict):
		plt.cla()
		plt.clf()
		color=cm.rainbow(np.linspace(0,1.0,5))
		markers = ['.','*','s','d','p']
		params=list(itertools.product(*[color,markers]))
		fig, ax = plt.subplots()
		ax.minorticks_on()
		for ck,m in zip(classdict.keys(),params[:len(classdict.keys())]):
			if classdict[ck].find('bulk')!=-1:
				if classdict[ck].find('fcc')!=-1:
					lilc=color[0]
				else:
					lilc=color[4]
				plt.scatter(pdf[pdf[2]==ck][0].values[::1000],pdf[pdf[2]==ck][1].values[::1000],marker='o',c=lilc,s=50,label=classdict[ck].translate(None,''.join(['_'])))
			elif classdict[ck].find('1')!=-1:
				#c=next(color)
				plt.scatter(pdf[pdf[2]==ck][0].values[:1000],pdf[pdf[2]==ck][1].values[:1000],marker=m[1],c=m[0],s=50,label=classdict[ck].translate(None,''.join(['_'])))
		plt.xlabel('PC1',fontsize=24)
		plt.ylabel('PC2', fontsize=24)
		plt.axis([xlo,xhi,ylo,yhi], fontsize=24)	
		plt.legend(bbox_to_anchor=(0.9,-0.15),prop={'size':12},ncol=4)
		plt.savefig('plotted.eps', edgecolor='none',bbox_inches='tight')


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
'''
if load_from_prev==False:
	
	# run the data through the pipeline
	alld,descriptors,DOUT=make_all_structures(se,0.04,clf_num)
	pickle.dump(alld, open( "alld"+str(clf_num)+".p", "wb" ) )
	# get pca object to save for later
	pca = PCA(n_components=se.num_comp)
	# dump the stats for each column, so they can be re-mapped in future simulations
	# NEED TO TURN THIS BACK ON TO WORK!!!!
	#dump_stats(alld)

	vals=list(np.unique(alld['desc'].values))
	keys=range(len(vals))
	classdict=dict(zip(keys,vals))
	rclassdict=dict(zip(vals,keys))
	# no more PCA is performed.
	newv,pca=consolidate_data(alld,pca,rclassdict)
	fdf=pd.DataFrame(newv)
	pickle.dump(fdf, open( "fdf"+str(clf_num)+".p", "wb"))
	pickle.dump(pca, open( "pca"+str(clf_num)+".p", "wb"))

else:

	fdf=pickle.load(open("fdf"+str(clf_num)+".p","rb"))
	alld=pickle.load(open("alld"+str(clf_num)+".p","rb"))
	pca=pickle.load(open("pca"+str(clf_num)+".p","rb"))

#turn the transformed bispec components into a df object
#now we make a dictionary of classifiers

vals=list(np.unique(alld['desc'].values))
keys=range(len(vals))

print '\n There are '+str(len(vals))+' options.'

classdict=dict(zip(keys,vals))
rclassdict=dict(zip(vals,keys))


pickle.dump(classdict, open( "classdict.p", "wb"))
'''
#assign final feature and descriptors
'''
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
	# should cross validate and train
	bispec_dict={1:2,2:5,3:8,4:14,5:20,6:30,7:40,8:55} #correlate the number of bispec comps. with max #
	clftotNN=KNeighborsClassifier(n_neighbors=5)
	clftot=RandomForestClassifier(n_estimators=1000, min_samples_leaf=10, verbose=True)
	X=fdf[fdf.columns[:bispec_dict[clf_num]]].values#[::100]
	Y=fdf[fdf.columns[bispec_dict[clf_num]]].values#[::100]

	#trim trains the samples with an even number from each sample (where possible)
	clftot,Xs,Ys,sc,Xv,Yv,AS=train_ML(clftot,X,Y,trim=True)
	clftotNN,Xs,Ys,sc,Xv,Yv,AS=train_ML(clftotNN,X,Y,trim=True)
	joblib.dump(clftot, './pickled/clftot'+str(clf_num)+'.pkl')
	joblib.dump(clftotNN, './pickled/clftotNN'+str(clf_num)+'.pkl')
'''

def plot_performance(max2j):
	fig, ax1 = plt.subplots()
	timing={1: 25.7669358253,2: 30.2913908958,3: 36.9177210331,4:48.7331619263,5: 65.2416930199,6: 95.2279620171,7: 137.776952982,8: 206.779714108}
	NNperf={8: 0.994687738005,7: 0.992707539985,6: 0.990936785986,5: 0.984799441483,4: 0.967485402386,3: 0.939610307185,2: 0.90834602691,1: 0.764229499}
	RFperf={1: 0.770938055344,2: 0.924682660574,3: 0.955185326225,4: 0.978585935517,5: 0.989902259457,6: 0.99487814166,7: 0.995760345265,8: 0.996566387408}
	ax1.plot(range(1,max2j+1),NNperf.values(),label='Nearest Neighbor')
	ax1.plot(range(1,max2j+1),RFperf.values(),label='Random Forest')
	ax1.legend(bbox_to_anchor=(1.0,0.8),prop={'size':12})
	ax1.set_xlabel('Max $2j$ value')
	ax1.set_ylabel('Fit Score')
	ax2 = ax1.twinx()
	ax2.plot(range(1,max2j+1),1-np.array(timing.values())/max(timing.values()),'--')
	ax2.set_ylabel('$1 - t/t_{max}$'+'(- -)')
	plt.savefig('performance.eps')
	pass
	

def make_variance_bias_data(max2j, NNmax, numsamples):
	
	testerror={}
	trainingerror={}	
	bispec_dict={1:2,2:5,3:8,4:14,5:20,6:30,7:40,8:55} #correlate the number of bispec comps. with max #
	
	for x in range(1,max2j+1):
		print '\n2j value ' + str(x)
		fdf=pickle.load(open("fdf"+str(x)+".p","rb"))
		X=fdf[fdf.columns[:bispec_dict[x]]].values
		Y=fdf[fdf.columns[bispec_dict[x]]].values
		testerror[x]=[]
		trainingerror[x]=[]
		# look at total number of NN used
		for NN in range(1,NNmax+1,2):
			print '\n For '+str(NN)+' nearest neighbors'
			temptest=[]
			temptrain=[]
			for nsamp in range(numsamples):		
				#clftotNN=KNeighborsClassifier(n_neighbors=NN)
				clftotNN=RandomForestClassifier(n_estimators=100, min_samples_leaf=NN)	
				clftotNN,Xs,Ys,sc,Xv,Yv,AS=train_ML(clftotNN,X,Y,trim=True,get_test_set=True,need_fit=True)
				temptest.append(AS)
				temptrain.append(sc)
			testerror[x].append(temptest)
			trainingerror[x].append(temptrain)
	return testerror,trainingerror



#TE,TRE=make_variance_bias_data(5,20,5)

#pickle.dump(TRE, open( "TRERF.p", "wb"))
#pickle.dump(TE, open( "TERF.p", "wb"))

#TRE=pickle.load(open("TRERF.p","rb"))
#TERF=pickle.load(open("TERF.p","rb"))

def plot_dat_variance(TE,TRE):
	fig, ax = plt.subplots()
	ax.minorticks_on()
	color=cm.rainbow(np.linspace(0,1.0,5))
	for x in TE:
		plt.plot(range(1,21,2),[np.array(TE[x][i]).mean() for i in range(len(TE[x]))],linewidth=3,c=color[x-1],label='$2j='+str(x)+'$'+' test')
		plt.plot(range(1,21,2),[np.array(TRE[x][i]).mean() for i in range(len(TRE[x]))],'--',linewidth=3,c=color[x-1],label='$2j='+str(x)+'$'+' training')	
	
	plt.xlabel('Minimum Samples Per Leaf',fontsize=24)
	plt.ylabel('Prediction Accuracy', fontsize=24)
	plt.axis([1,19,0.6,1.0], fontsize=24)	
	plt.legend(bbox_to_anchor=(1.1,-0.11),prop={'size':18},ncol=3)	
	#plt.legend()
	plt.savefig('Variance.eps', edgecolor='none',bbox_inches='tight')


