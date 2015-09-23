import random
class template_gen:
	
	def __init__(self,temp_val=0.04,bispec_comp=8):
		self.bispec_comp=bispec_comp
		self.temp_val=0.04
		bispec_dict={1:2,2:5,3:8,4:14,5:20,6:30,7:40,8:55}
		def adjust_temp_read(lat,output,datname, athermal=False,tv=temp_val, bc=bispec_comp):
			if athermal==False:
				out='units metal\nboundary p p p\nread_data '+str(datname)+'\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+ 'displace_atoms all random '+str(tv)+' '+str(tv)+' '+str(tv)+' '+str(random.randint(10,1000))+'\n'+\
				'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 '+str(bc)+ ' '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z '
				for t in range(1,bispec_dict[bc]+1):
					out = out+'c_vb['+str(t)+'] '
				out=out+'\n'
			else:
				out='units metal\nboundary p p p\nread_data '+str(datname)+'\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+\
				'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 '+str(bc)+' '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z '
				for t in range(1,bispec_dict[bc]+1):
					out = out+'c_vb['+str(t)+'] '
				out=out+'\n'
			
			return out  

		def adjust_temp(lat,lx,ly,lz,xi,yi,zi,bound,struct,output,tv=temp_val, bc=bispec_comp):

			out='units metal\nboundary        '+str(bound)+'\nregion 		sim block -'+str(lx*3)+' '+str(lx*2)+' -'+str(ly*3)+' '+str(ly*2)+' -'+str(lz*3)+' '+str(lz*2)+'\n'+'create_box 1 sim\n'+ \
				'lattice '+struct+' '+str(lat)+' origin 0 0 0 orient x '+str(xi[0])+' '+str(xi[1])+' '+str(xi[2])+' orient y '+ str(yi[0])+' '+str(yi[1])+' '+str(yi[2])+' orient z '+str(zi[0])+' '+str(zi[1])+' '+str(zi[2])+ '\n'+'create_atoms 1 box\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+ 'displace_atoms all random '+str(tv)+' '+str(tv)+' '+str(tv)+' '+str(random.randint(10,1000))+'\n'+\
				'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 '+str(bc)+' '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z '

			for t in range(1,bispec_dict[bc]+1):
				out = out+'c_vb['+str(t)+'] '
			out=out+'\n'
			
			return out

		def adjust_temp_athermal(lat,lx,ly,lz,xi,yi,zi,bound,struct,output, bc=bispec_comp):

			out='units metal\nboundary        '+str(bound)+'\nregion 		sim block -'+str(lx*3)+' '+str(lx*2)+' -'+str(ly*3)+' '+str(ly*2)+' -'+str(lz*3)+' '+str(lz*2)+'\n'+'create_box 1 sim\n'+ \
				'lattice '+struct+' '+str(lat)+' origin 0 0 0 orient x '+str(xi[0])+' '+str(xi[1])+' '+str(xi[2])+' orient y '+ str(yi[0])+' '+str(yi[1])+' '+str(yi[2])+' orient z '+str(zi[0])+' '+str(zi[1])+' '+str(zi[2])+ '\n'+'create_atoms 1 box\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+\
				'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 '+str(bc)+' '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z '
			
			for t in range(1,bispec_dict[bc]+1):
				out = out+'c_vb['+str(t)+'] '
			out=out+'\n'
			
			return out


		self.adjust_temp_read=adjust_temp_read
		self.adjust_temp=adjust_temp
		self.adjust_temp_athermal=adjust_temp_athermal



# for testing purposes
if __name__=='__main__':
	x=template_gen(temp_val=10.90, bispec_comp=2)
	look=x.adjust_temp_read(1,'2','2')
