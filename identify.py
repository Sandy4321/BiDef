#this script identifies the entries coming from a lammps dump -- any lammps dump using the specified trained ML classifier
#OK well this works good enough for a prototype with just surfaces... does it work for vacancies???!


import pickle
import pandas as pd
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn import svm

def norm (s, m) : return lambda x : (x - m) / s

KM=KMeans(n_clusters=2)

def nab_and_format_bispec(fn):
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


	#atom_cats=KM.fit_predict(trans_values)
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

dat,output,trans,tdata,probs=nab_and_format_bispec("example_fcc")

make_output("example_fcc",dat,output)
