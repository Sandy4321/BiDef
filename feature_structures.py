import random
import numpy as np
from lammps import lammps
import pandas as pd
import itertools
import subprocess
from ovito_master import *

def get_boxes(scale):
	b=['x scale '+str(scale), ' y scale '+str(scale), ' z scale '+str(scale)]
	return [b[0],b[1],b[2], b[0]+b[1],b[0]+b[2],b[1]+b[2],b[0]+b[1]+b[2]]

def get_boxesf(scale,f):
	b=['x scale '+str(scale*f), ' y scale '+str(scale*f), ' z scale '+str(scale*f)]
	return [b[0],b[1],b[2], b[0]+b[1],b[0]+b[2],b[1]+b[2],b[0]+b[1]+b[2]]

def adjust_temp(lat,lx,ly,lz,xi,yi,zi,bound,struct,output):

	return 'units metal\nboundary        '+str(bound)+'\nregion 		sim block -'+str(lx*3)+' '+str(lx*2)+' -'+str(ly*3)+' '+str(ly*2)+' -'+str(lz*3)+' '+str(lz*2)+'\n'+'create_box 1 sim\n'+ \
		'lattice '+struct+' '+str(lat)+' origin 0 0 0 orient x '+str(xi[0])+' '+str(xi[1])+' '+str(xi[2])+' orient y '+ str(yi[0])+' '+str(yi[1])+' '+str(yi[2])+' orient z '+str(zi[0])+' '+str(zi[1])+' '+str(zi[2])+ '\n'+'create_atoms 1 box\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+ 'displace_atoms all random 0.05 0.05 0.05 '+str(random.randint(10,1000))+'\n'+\
		'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 8 '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z c_vb[1] c_vb[2] c_vb[3] c_vb[4] c_vb[5] c_vb[6] c_vb[7] c_vb[8] c_vb[9] c_vb[10] c_vb[11] c_vb[12] c_vb[13] c_vb[14] c_vb[15] c_vb[16] c_vb[17] c_vb[18] c_vb[19] c_vb[20] c_vb[21] c_vb[22] c_vb[23] c_vb[24] c_vb[25] c_vb[26] c_vb[27] c_vb[28] c_vb[29] c_vb[30] c_vb[31] c_vb[32] c_vb[33] c_vb[34] c_vb[35] c_vb[36] c_vb[37] c_vb[38] c_vb[39] c_vb[40] c_vb[41] c_vb[42] c_vb[43] c_vb[44] c_vb[45] c_vb[46] c_vb[47] c_vb[48] c_vb[49] c_vb[50] c_vb[51] c_vb[52] c_vb[53] c_vb[54] c_vb[55]\n'

def adjust_temp_athermal(lat,lx,ly,lz,xi,yi,zi,bound,struct,output):

	return 'units metal\nboundary        '+str(bound)+'\nregion 		sim block -'+str(lx*3)+' '+str(lx*2)+' -'+str(ly*3)+' '+str(ly*2)+' -'+str(lz*3)+' '+str(lz*2)+'\n'+'create_box 1 sim\n'+ \
		'lattice '+struct+' '+str(lat)+' origin 0 0 0 orient x '+str(xi[0])+' '+str(xi[1])+' '+str(xi[2])+' orient y '+ str(yi[0])+' '+str(yi[1])+' '+str(yi[2])+' orient z '+str(zi[0])+' '+str(zi[1])+' '+str(zi[2])+ '\n'+'create_atoms 1 box\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+\
		'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 8 '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z c_vb[1] c_vb[2] c_vb[3] c_vb[4] c_vb[5] c_vb[6] c_vb[7] c_vb[8] c_vb[9] c_vb[10] c_vb[11] c_vb[12] c_vb[13] c_vb[14] c_vb[15] c_vb[16] c_vb[17] c_vb[18] c_vb[19] c_vb[20] c_vb[21] c_vb[22] c_vb[23] c_vb[24] c_vb[25] c_vb[26] c_vb[27] c_vb[28] c_vb[29] c_vb[30] c_vb[31] c_vb[32] c_vb[33] c_vb[34] c_vb[35] c_vb[36] c_vb[37] c_vb[38] c_vb[39] c_vb[40] c_vb[41] c_vb[42] c_vb[43] c_vb[44] c_vb[45] c_vb[46] c_vb[47] c_vb[48] c_vb[49] c_vb[50] c_vb[51] c_vb[52] c_vb[53] c_vb[54] c_vb[55]\n'

def adjust_temp_read(lat,output,datname):

	return 'units metal\nboundary s s p\nread_data '+str(datname)+'\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+ 'displace_atoms all random 0.05 0.05 0.05 '+str(random.randint(10,1000))+'\n'+\
		'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 8 '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z c_vb[1] c_vb[2] c_vb[3] c_vb[4] c_vb[5] c_vb[6] c_vb[7] c_vb[8] c_vb[9] c_vb[10] c_vb[11] c_vb[12] c_vb[13] c_vb[14] c_vb[15] c_vb[16] c_vb[17] c_vb[18] c_vb[19] c_vb[20] c_vb[21] c_vb[22] c_vb[23] c_vb[24] c_vb[25] c_vb[26] c_vb[27] c_vb[28] c_vb[29] c_vb[30] c_vb[31] c_vb[32] c_vb[33] c_vb[34] c_vb[35] c_vb[36] c_vb[37] c_vb[38] c_vb[39] c_vb[40] c_vb[41] c_vb[42] c_vb[43] c_vb[44] c_vb[45] c_vb[46] c_vb[47] c_vb[48] c_vb[49] c_vb[50] c_vb[51] c_vb[52] c_vb[53] c_vb[54] c_vb[55]\n'


# we need to nab the bispec components for every defect type, then merge them all into one dataframe, then standardize the coefficients, then perform the PCA

def nab_bispec_train(infile):
	df=pd.read_csv(infile,skiprows=8,delim_whitespace=True,low_memory=False)
	# get rid of the bogus first columns
	cols=df.columns[2::]
	df.drop(df.columns[-2::],1,inplace=True)
	df.columns=cols
	return df

#need one of every surface type and of each structure -- then need to apply the 
#prototype to EACH one of the systems

def make_surface_prototypes(se,alld,dic):
	DOUT={}
	tdict={(se.structs[0],tuple(se.indicies[0][0])):'fcc_111',\
			(se.structs[0],tuple(se.indicies[0][1])):'fcc_110',\
			(se.structs[0],tuple(se.indicies[0][2])):'fcc_112',\
			(se.structs[0],tuple(se.indicies[1][2])):'fcc_100',\
			(se.structs[1],tuple(se.indicies[0][0])):'bcc_111',\
			(se.structs[1],tuple(se.indicies[0][1])):'bcc_110',\
			(se.structs[1],tuple(se.indicies[0][2])):'bcc_112',\
			(se.structs[1],tuple(se.indicies[1][2])):'bcc_100'}	
	counter=0
	for stru in se.structs:
			for i in se.indicies:
				for bound in se.bounds:
						if i[dic[bound]]==[0,1,0] or i[dic[bound]]==[0,0,1]:
							break

						lx=se.lattice*np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
						ly=se.lattice*np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
						lz=se.lattice*np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)							
						
						fiz='tempstruct_'+str(counter)
						counter+=1
						template=adjust_temp_athermal(se.lattice,lx,ly,lz,i[0],i[1],i[2],bound,stru,fiz)

						with open('temp.in', 'w') as f:
								f.write(template)
								f.write('\nrun 0 post no\n')
						lmp=lammps()
						try:							
							lmp.file('temp.in')
							temp=nab_bispec_train(fiz)
							alld=alld.append(temp)
						except:
							print "\nFAILURE TO RUN LAMMPS."																					
						lmp.close()
						#get the CNA parameters from ovito
						check=run_ovitos('CNA',fiz,fiz+'CNAOUT')
						lil_d={fiz+'CNAOUT':tdict[(stru,tuple(i[dic[bound]]))]}
						if stru=='fcc':
							lil_cond={fiz+'CNAOUT':['CNA',1]}
						elif stru=='bcc':
							lil_cond={fiz+'CNAOUT':['CNA',3]}
						DOUT[(stru,tuple(i[dic[bound]]))]=initialize_dislocation_descriptors(fiz+'CNAOUT',lil_d,lil_cond)
						#reset the index to the atomic IDs for the merge with the bispec data
						DOUT[(stru,tuple(i[dic[bound]]))].index=DOUT[(stru,tuple(i[dic[bound]]))]['PID']
							#need_proto=False
	print "\n\n\n			FINISHED SURFACE PROTOTYPES		\n\n\n"
	return DOUT						

def surfaces(se,alld,descriptors,dic,DOUT):
	
	need_proto=False
	#DOUT={}

	for stru in se.structs:
			for i in se.indicies:
				for bound in se.bounds:
					for s in se.scales:
						box=get_boxes(s)
						for b in box:
							# we only need to check one [100] face!!
							if i[dic[bound]]==[0,1,0] or i[dic[bound]]==[0,0,1]:
								break		
							for t in range(se.thermal):

								lx=se.lattice*np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
								ly=se.lattice*np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
								lz=se.lattice*np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)

								if random.random() < se.sampling_rate:
									fiz='example'+'_'+stru+str(random.randint(0,10000))
								else:
									fiz='temporaryfile'

								template=adjust_temp(se.lattice,lx,ly,lz,i[0],i[1],i[2],bound,stru,fiz)
		
								with open('temp.in', 'w') as f:
									f.write(template)
									f.write('\nchange_box all '+b+' boundary '+bound+' remap units box\n')
									f.write('\nrun 0 post no\n')
				
								lmp=lammps()
								try:							
									lmp.file('temp.in')
									temp=nab_bispec_train(fiz)
									#alld=alld.append(temp)
								except:
									print "\nFAILURE TO RUN LAMMPS."																					
								lmp.close()
								# we need the number of atoms, the bounds (to tell which is periodic), the orientations, and the structure
								#descriptors.append([i,s,b,t,bound,len(temp)])
								if i == se.indicies[1]:
									strang='1 0 0'
								else:
									strang=str(i[dic[bound]])
								print '\n'+strang+' '+stru
								temp.index=temp['id']
								temp['desc']=DOUT[(stru,tuple(i[dic[bound]]))]['desc']

								alld=alld.append(temp)
	return alld


def make_vacancy_prototypes(se,alld,dic):
	#dictionary of prototype features	
	DOUT={}
	
	dcheck={se.structs[0]:False,se.structs[1]:False}
	tdict={se.structs[0]:'fcc_vacancy',se.structs[1]:'bcc_vacancy'}
	counter=0
	i = se.indicies[1] #standard cell
	for stru in se.structs:

				lx=se.lattice*np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
				ly=se.lattice*np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
				lz=se.lattice*np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)							
				
				fiz='tempstruct_'+str(counter)
				counter+=1
				template=adjust_temp_athermal(se.lattice,lx,ly,lz,i[0],i[1],i[2],'p p p',stru,fiz)

				with open('temp.in', 'w') as f:
						f.write(template)
						f.write('\n group g1 id 1')
						f.write('\n delete_atoms group g1')
						f.write('\nrun 0 post no\n')
				lmp=lammps()
				try:							
					lmp.file('temp.in')
					temp=nab_bispec_train(fiz)
					#alld=alld.append(temp)
				except:
					print "\nFAILURE TO RUN LAMMPS."																					
				lmp.close()
				#make a prototype if needed
				#if need_proto==True:
				check=run_ovitos('CNA',fiz,fiz+'CNAOUT')
				lil_d={fiz+'CNAOUT':tdict[stru]}
				if stru=='fcc':
					lil_cond={fiz+'CNAOUT':['CNA',1]}
				elif stru=='bcc':
					lil_cond={fiz+'CNAOUT':['CNA',3]}
				DOUT[stru]=initialize_dislocation_descriptors(fiz+'CNAOUT',lil_d,lil_cond)
				#reset the index to the atomic IDs for the merge with the bispec data
				DOUT[stru].index=DOUT[stru]['PID']
					#need_proto=False
	print "\n\n\n			FINISHED VACANCY PROTOTYPES		\n\n\n"
	return DOUT	

def vacancies(se, alld, descriptors,DOUT):

#loops to create all of the vacancies
	for stru in se.structs:
		for s in se.scales:
			box=get_boxes(s)
			for b in box:
				for t in range(se.thermal):

					lx=se.lattice
					ly=se.lattice
					lz=se.lattice

					if random.random() < se.sampling_rate:
						fiz='vacancy_'+stru+str(random.randint(0,10000))
					else:
						fiz='temporaryfile'

					template=adjust_temp(se.lattice,lx,ly,lz,[1,0,0],[0,1,0],[0,0,1],'p p p',stru,fiz)

					with open('temp.in', 'w') as f:
						f.write(template)
						f.write('\n group g1 id 1')
						f.write('\n delete_atoms group g1')
						f.write('\nchange_box all '+b+' boundary '+'p p p'+' remap units box\n')
						f.write('\nrun 0 post no\n')
	
					lmp=lammps()
					try:					
						lmp.file('temp.in')
						temp=nab_bispec_train(fiz)
						#alld=alld.append(temp)
					except:
						print "\nFAILURE TO RUN LAMMPS."						
					lmp.close()
			
					# we need the number of atoms, the structure, flag that it's a vacancy
					print '\n'+'vacancy '+stru
					temp.index=temp['id']
					temp['desc']=DOUT[stru]['desc']

					alld=alld.append(temp)

	return alld


def make_interstitial_prototypes(se,alld,dic):
	#dictionary of prototype features	
	DOUT={}
	ot={0:'tetrahedral',1:'octahedral'}
	tdict={(se.structs[0],ot[0]):'fcc_tet_interstitial',\
		   (se.structs[0],ot[1]):'fcc_oct_interstitial',\
		   (se.structs[1],ot[0]):'bcc_tet_interstitial',\
		   (se.structs[1],ot[1]):'bcc_oct_interstitial'}
	counter=0
	i = se.indicies[1] #standard cell


	for stru in se.structs:
		c=0
		for inter in se.interstitial[stru]:
			lx=se.lattice*np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
			ly=se.lattice*np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
			lz=se.lattice*np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)							
			
			fiz='tempstruct_'+str(counter)
			counter+=1
			template=adjust_temp_athermal(se.lattice,lx,ly,lz,i[0],i[1],i[2],'p p p',stru,fiz)

			with open('temp.in', 'w') as f:
				f.write(template)
				f.write('\ncreate_atoms 1 single '+str(inter[0])+' '+str(inter[1])+' '+str(inter[2]))
				f.write('\nrun 0 post no\n')
			lmp=lammps()
			try:							
				lmp.file('temp.in')
				temp=nab_bispec_train(fiz)
			except:
				print "\nFAILURE TO RUN LAMMPS."																					
			lmp.close()
			#make a prototype if needed
			check=run_ovitos('CNA',fiz,fiz+'CNAOUT')
			lil_d={fiz+'CNAOUT':tdict[(stru,ot[c])]}
			if stru=='fcc':
				lil_cond={fiz+'CNAOUT':['CNA',1]}
			elif stru=='bcc':
				lil_cond={fiz+'CNAOUT':['CNA',3]}
			DOUT[(stru,ot[c])]=initialize_dislocation_descriptors(fiz+'CNAOUT',lil_d,lil_cond)
			#reset the index to the atomic IDs for the merge with the bispec data
			DOUT[(stru,ot[c])].index=DOUT[(stru,ot[c])]['PID']
			c+=1
					#need_proto=False
	print "\n\n\n			FINISHED INTERSTITIAL PROTOTYPES		\n\n\n"
	return DOUT


def interstitials(se,alld,descriptors,DOUT):

	ot={0:'tetrahedral',1:'octahedral'}
	for stru in se.structs:
		c=0
		for inter in se.interstitial[stru]:
			for s in se.scales:
				box=get_boxes(s)
				for b in box:
					for t in range(se.thermal):
						
						lx=se.lattice
						ly=se.lattice
						lz=se.lattice
						
						if random.random() < se.sampling_rate:
							fiz='keep_'+ot[c]+'interstitial_'+stru+str(random.randint(0,10000))
						else:
							fiz='temporaryfile'

						template=adjust_temp(se.lattice,lx,ly,lz,[1,0,0],[0,1,0],[0,0,1],'p p p',stru,fiz)

						with open('temp.in', 'w') as f:
							f.write(template)
							f.write('\ncreate_atoms 1 single '+str(inter[0])+' '+str(inter[1])+' '+str(inter[2]))
							f.write('\nchange_box all '+b+' boundary '+'p p p'+' remap units box\n')
							f.write('\nrun 0 post no\n')

						lmp=lammps()
						try:						
							lmp.file('temp.in')
							temp=nab_bispec_train(fiz)
							#alld=alld.append(temp)
						except:
							print "\nFAILURE TO RUN LAMMPS."						
						lmp.close()
						# we need the number of atoms, the structure, flag for the type of intersitial
						print '\n'+ot[c]+'interstitial '+stru
						temp.index=temp['id']
						temp['desc']=DOUT[(stru,ot[c])]['desc']

						alld=alld.append(temp)
			c+=1

	return alld

def make_dislocations_prototypes(se):
	#OK what types of defects, FCC full, partial, BCC full... strain and thermalize
	DOUT={}
	tdict={se.defect_names[0]:'fcc_full_disloc',\
		   se.defect_names[1]:'bcc_full_disloc',\
		   se.defect_names[2]:'fcc_partial_disloc'}
	
	for d in se.defect_names:
		lil_d={se.flocs[d]:tdict[d]}
		if tdict[d].find('fcc')!=-1:
			if tdict[d].find('full')!=-1:
				lil_cond={se.flocs[d]:['CSP_surf',1]}
			else:
				lil_cond={se.flocs[d]:['CNA',1]}
		elif tdict[d].find('bcc')!=-1:
			lil_cond={se.flocs[d]:['CNA_surf',3]}

		DOUT[d]=initialize_dislocation_descriptors(se.flocs[d],lil_d,lil_cond)	
		DOUT[d].index=DOUT[d]['PID']
	print "\n\n\n			FINISHED DISLOCATION PROTOTYPES		\n\n\n"
	return DOUT



# make system calls to atomsk -- easy way to generate the inital dislocation stuctures, then strain and thermalize them such that they are still identified.
def dislocations(se,alld,descriptors,DOUT):
	#OK what types of defects, FCC full, partial, BCC full... strain and thermalize
	for d in se.defect_names:
		for s in se.scales:
			if d.find('full')!=-1 or d.find('bcc')!=-1:
				box=get_boxesf(s,se.lattice/4.02)
			else:
				box=get_boxes(s)
			for b in box:
				for t in range(se.thermal):
						
						lx=se.lattice
						ly=se.lattice
						lz=se.lattice
						
						if random.random() < se.sampling_rate:
							fiz='dislocation'+d+str(random.randint(0,10000))
						else:
							fiz='temporaryfile'

						template=adjust_temp_read(se.lattice,fiz,d)

						with open('temp.in', 'w') as f:
							f.write(template)
							f.write('\nchange_box all '+b+' boundary '+'p p p'+' remap units box\n')
							f.write('\nrun 0 post no\n')

						lmp=lammps()
						try:						
							lmp.file('temp.in')
							temp=nab_bispec_train(fiz)
							#alld=alld.append(temp)
						except:
							print "\nFAILURE TO RUN LAMMPS."						
						lmp.close()
						# we need the number of atoms, the structure, flag for the type of intersitial
						print '\n'+d+'dislocation '
						temp.index=temp['id']
						temp['desc']=DOUT[d]['desc']
						alld=alld.append(temp)
	return alld





#assign the descriptors for each of the dislocation structures (no Kmeans necessary)
def initialize_dislocation_descriptors(f,tdict,condit):
	#f has the each of the files to consider, tdict relates the file name to the defect name
	#condit contains the conditions for each simulation, surf or no surf, CNA or CSP [0] spot
	# is the description and the [1] spot is the value needed
	tdf=pd.read_csv(f,delim_whitespace=True)
	tdf['desc']='Please Drop'
	if condit[f][0].find('surf')!=-1:#if it needs to trim the surfaces
		#filter out the surface atoms
		t1=tdf[tdf['X']<tdf['X'].max()-3.]
		t1=t1[t1['X']>t1['X'].min()+3.]
		t1=t1[t1['Y']<t1['Y'].max()-3.]
		t1=t1[t1['Y']>t1['Y'].min()+3.]
		if condit[f][0].find('CSP')!=-1:
			t1.ix[t1[t1['TYPE']>condit[f][1]].index,'desc']=tdict[f]
			if tdict[f].find('fcc')!=-1:
				t1.ix[t1[t1['TYPE']<=condit[f][1]].index,'desc']='bulk fcc'
			else:
				t1.ix[t1[t1['TYPE']<=condit[f][1]].index,'desc']='bulk bcc'
		elif condit[f][0].find('CNA')!=-1:
			t1.ix[t1[t1['TYPE']!=condit[f][1]].index,'desc']=tdict[f]
			if tdict[f].find('fcc')!=-1:
				t1.ix[t1[t1['TYPE']==condit[f][1]].index,'desc']='bulk fcc'
			else:
				t1.ix[t1[t1['TYPE']==condit[f][1]].index,'desc']='bulk bcc'
		
		tdf.ix[t1.index,'desc']=t1['desc']
	else:
		if condit[f][0].find('CNA')!=-1:
			tdf.ix[tdf[tdf['TYPE']!=condit[f][1]].index,'desc']=tdict[f]
			if tdict[f].find('fcc')!=-1:
				tdf.ix[tdf[tdf['TYPE']==condit[f][1]].index,'desc']='bulk fcc'
			else:
				tdf.ix[tdf[tdf['TYPE']==condit[f][1]].index,'desc']='bulk bcc'
		elif condit[f][0].find('CSP')!=-1:
			tdf.ix[tdf[tdf['TYPE']>condit[f][1]].index,'desc']=tdict[f]
			if tdict[f].find('fcc')!=-1:
				tdf.ix[tdf[tdf['TYPE']<=condit[f][1]].index,'desc']='bulk fcc'
			else:
				tdf.ix[tdf[tdf['TYPE']<=condit[f][1]].index,'desc']='bulk bcc'
	
	return tdf


def make_all_structures(se):

	alld=pd.DataFrame()
	dic={se.bounds[0]:2,se.bounds[1]:1,se.bounds[2]:0} #convert the different boundaries to an index
	descriptors=[]
	#Bi spectrum gathering pipeline
	DOUT=make_surface_prototypes(se,alld,dic)
	alld=surfaces(se,alld,descriptors,dic,DOUT)
	DOUT=make_vacancy_prototypes(se,alld,dic)
	alld=vacancies(se,alld,descriptors,DOUT)
	DOUT=make_interstitial_prototypes(se,alld,dic)
	alld=interstitials(se,alld,descriptors,DOUT)
	DOUT=make_dislocations_prototypes(se)
	alld=dislocations(se,alld,descriptors,DOUT)
	#may not work, may redo above with better organization of alld
	alld.index = range(len(alld))	
	alld=alld.dropna()
	alld=alld[alld['desc']!='Please Drop']
	
	return alld,descriptors,DOUT

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
	 self.flocs={self.defect_names[0]:'fccfull_temp.structures',self.defect_names[1]:'BCC_temp.structures',self.defect_names[2]:'temp.structures'} #dislocation prototype file structural analysis files
	
	
#f has the each of the files to consider, tdict relates the file name to the defect name
	#condit contains the conditions for each simulation, surf or no surf, CNA or CSP [0] spot
	# is the description and the [1] spot is the value needed

se=simulation_env()

alld,descriptors,DOUT=make_all_structures(se)


