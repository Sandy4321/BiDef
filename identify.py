#this script identifies the entries coming from a lammps dump -- any lammps dump using the specified trained ML classifier



import pickle
import pandas as pd
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np

def norm (s, m) : return lambda x : (x - m) / s


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


def nab_and_format_bispec(fn,get_cats=False):
	df=pd.read_csv(fn,skiprows=8,delim_whitespace=True,low_memory=False)
	# get rid of the bogus first columns
	cols=df.columns[2::]
	df.drop(df.columns[-2::],1,inplace=True)
	df.columns=cols
	
	tdata=df

	#Need the stats for each bispec component
	print "\nre-normalizing data..."
	for x in tdata.columns[5::]:
		m=pickle.load(open( "./data_stats/"+str(x)+"_m.p", "rb"))
		s=pickle.load(open( "./data_stats/"+str(x)+"_s.p", "rb"))
		tdata[x]=tdata[x].map(norm(s,m))
	



	# now we transform it using the previously trained PCA
	print "\nloading in PCA..."
	pca=pickle.load(open("pca.p","rb"))
	trans_values=pca.transform(tdata[tdata.columns[5::]].values)
	if get_cats==True:
		KM=KMeans(n_clusters=2)
		print "\n Separating into " + str(KM.get_params()['n_clusters']) +" parts, and removing surface atoms."
		nonsurf=remove_surface_atoms(df)
		atom_cats=np.zeros(len(trans_values))
		atom_cats[nonsurf]=KM.fit_predict(trans_values[nonsurf])
		return df,atom_cats
	#make_output(fn,df,atom_cats)

	print "\n loading in classifier..."
	clf=joblib.load('./pickled/rf.pkl')
	print "\n making prediction..."
	out=clf.predict(trans_values)
	from scipy import stats
	mode=stats.mode(out)
	out2=clf.predict_proba(trans_values)
	
	#only identify above 75% certainty, otherwise, go to the mode.
#	for o,c in zip(out2,range(len(out))):
#		if out[c] != mode[0] and sum(o>0.75)!=0:	
#			out[c]=(o>0.75).argmax()
#		else:
#			out[c]=mode[0]
		
			

	return df,out,trans_values,tdata,out2


def make_output(fn,df,out):

	with open(fn) as myfile:
	    head = [next(myfile) for x in xrange(8)]

	with open('output', 'w') as f:
		for h in head:    
			f.write(h) 
		f.write('ITEM: ATOMS id type x y z ') 
		f.write('vid')
		f.write('\n')
		for x in range(len(df)):
			for v in df[x:x+1].values[0][:5]:
				f.write(str(v)+' ')
			f.write(str(out[x]))
			f.write('\n')
	pass

if __name__ == "__main__":

	dat,output,trans,tdata,probs=nab_and_format_bispec("dislocationbcc_disloc.lmp6538")
	#dat,output=nab_and_format_bispec("dislocationbcc_disloc.lmp6538",get_cats=True)	
	make_output("dislocationbcc_disloc.lmp6538",dat,output)

