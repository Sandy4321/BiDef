# get the num NN for each defective particle, printed with ML ID

import sys
from ovito.io import *
from ovito.modifiers import *
from ovito.anim import *
from ovito.data import CutoffNeighborFinder
from ovito.modifiers import *


node = None
cutoff = 3.0
#stylename='Structure Type'
#num_style=3
stylename='vid'
num_style=10


for file in sys.argv[1:]:

	if not node:
		# Import the first file using import_file().
		# This creates the ObjectNode and sets up the modification pipeline.
		node = import_file(file,multiple_frames=True)
		print(node.source.num_frames)
		#cna = CommonNeighborAnalysisModifier(adaptive_mode=True)
		mod = SelectExpressionModifier(expression ="vid!=15")
		node.modifiers.append(mod)
		dmod=DeleteSelectedParticlesModifier()
		node.modifiers.append(dmod)
		histox=HistogramModifier(bin_count=20,property="Position.X")
		node.modifiers.append(histox)
		histoy=HistogramModifier(bin_count=20,property="Position.Y")
		node.modifiers.append(histoy)
		histoz=HistogramModifier(bin_count=20,property="Position.Z")
		node.modifiers.append(histoz)
	else:
	        continue	

# Evaluate pipeline and wait until the analysis results are available.
c=[]

fout=open(sys.argv[2:][0],'w')
fout.write('Frame Hx Hy Hz\n')
for frame in range(0, ovito.dataset.anim.last_frame + 1):
	errors=[]
	NNe=[]
	ovito.dataset.anim.current_frame = frame    # Jump to the animation frame.
	o=node.compute()
		
	Hx=histox.histogram
	Hy=histoy.histogram
	Hz=histoz.histogram

	spacingx=Hx[1][0]-Hx[0][0]
	spacingy=Hy[1][0]-Hy[0][0]
	spacingz=Hz[1][0]-Hz[0][0]

	
	for hx,hy,hz in zip(Hx,Hy,Hz):
		fout.write(str(frame)+' '+str(hx[1])+' '+str(hy[1])+' '+str(hz[1])+' '+str(spacingx)+' '+str(spacingy)+' '+str(spacingz)+'\n')

fout.close()
