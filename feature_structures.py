import random
import numpy as np
from lammps import lammps
import pandas as pd

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



def adjust_temp_read(lat,output,datname):

	return 'units metal\nread_data '+str(datname)+'\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+ 'displace_atoms all random 0.05 0.05 0.05 '+str(random.randint(10,1000))+'\n'+\
		'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 8 '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z c_vb[1] c_vb[2] c_vb[3] c_vb[4] c_vb[5] c_vb[6] c_vb[7] c_vb[8] c_vb[9] c_vb[10] c_vb[11] c_vb[12] c_vb[13] c_vb[14] c_vb[15] c_vb[16] c_vb[17] c_vb[18] c_vb[19] c_vb[20] c_vb[21] c_vb[22] c_vb[23] c_vb[24] c_vb[25] c_vb[26] c_vb[27] c_vb[28] c_vb[29] c_vb[30] c_vb[31] c_vb[32] c_vb[33] c_vb[34] c_vb[35] c_vb[36] c_vb[37] c_vb[38] c_vb[39] c_vb[40] c_vb[41] c_vb[42] c_vb[43] c_vb[44] c_vb[45] c_vb[46] c_vb[47] c_vb[48] c_vb[49] c_vb[50] c_vb[51] c_vb[52] c_vb[53] c_vb[54] c_vb[55]\n'


# we need to nab the bispec components for every defect type, then merge them all into one dataframe, then standardize the coefficients, then perform the PCA

def nab_bispec_train(infile):
	df=pd.read_csv(infile,skiprows=8,delim_whitespace=True,low_memory=False)
	# get rid of the bogus first columns
	cols=df.columns[2::]
	df.drop(df.columns[-2::],1,inplace=True)
	df.columns=cols
	return df


def surfaces(se,alld,descriptors,dic):
	for stru in se.structs:
			for i in se.indicies:
				for bound in se.bounds:
					for s in se.scales:
						box=get_boxes(s)
						for b in box:
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
									alld=alld.append(temp)
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
								descriptors.append([strang+' '+stru,len(temp)])
	return alld, descriptors



def vacancies(se, alld, descriptors):

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
						alld=alld.append(temp)
					except:
						print "\nFAILURE TO RUN LAMMPS."						
					lmp.close()

					# we need the number of atoms, the structure, flag that it's a vacancy
					print '\n'+'vacancy '+stru
					descriptors.append(['vacancy '+stru,len(temp)])

	return alld, descriptors


def interstitials(se,alld,descriptors):

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
							f.write('\ncreate_atoms 1 single '+str(se.inter[0])+' '+str(se.inter[1])+' '+str(se.inter[2]))
							f.write('\nchange_box all '+b+' boundary '+'p p p'+' remap units box\n')
							f.write('\nrun 0 post no\n')

						lmp=lammps()
						try:						
							lmp.file('temp.in')
							temp=nab_bispec_train(fiz)
							alld=alld.append(temp)
						except:
							print "\nFAILURE TO RUN LAMMPS."						
						lmp.close()
						# we need the number of atoms, the structure, flag for the type of intersitial
						print '\n'+ot[c]+'interstitial '+stru
						descriptors.append([ot[c]+'interstitial '+stru,len(temp)])
			c+=1

	return alld, descriptors

# make system calls to atomsk -- easy way to generate the inital dislocation stuctures, then strain and thermalize them such that they are still identified.
def partial_dislocations(se,alld,descriptors):
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
							alld=alld.append(temp)
						except:
							print "\nFAILURE TO RUN LAMMPS."						
						lmp.close()
						# we need the number of atoms, the structure, flag for the type of intersitial
						print '\n'+d+'dislocation '
						descriptors.append([d+'disloc ',len(temp)])
	return alld, descriptors



def make_all_structures(se):

	alld=pd.DataFrame()
	dic={se.bounds[0]:0,se.bounds[1]:1,se.bounds[2]:2} #convert the different boundaries to an index
	descriptors=[]
	#Bi spectrum gathering pipeline
	#alld,desctriptors=surfaces(se,alld,descriptors,dic)
	#alld,desctriptors=vacancies(se,alld,descriptors)
	#alld,desctriptors=intersitials(se,alld,descriptors)
	alld,descriptors=partial_dislocations(se,alld,descriptors)
	#reorganize the descriptors
	d2=[]
	for d in descriptors:
		for t in range(d[1]):
			d2.append(d)
	
	alld['desc'] = [d[0] for d in d2]
	
	return alld


