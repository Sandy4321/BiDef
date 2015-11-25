# BiDef
Machine learning code and feature generation for crystal defect identification


The code generates all training data and  trains a  dataset based  on parameters entered into generate_defect_structs.py
It is based on having several lammps dump templates constructed with the defects of interest, and the defective
atoms are then extracted using either  CSP or CNA for training. After the data is generated it can be either saved
or loaded in generate_defect_structs, where the actual training occurs. After the  classification algorithms are trained
they are saved in an extrernal folder for later use.

Each new  identification is performed with the identify.py file. The command line parameters given to the file
tell it what to identify with which ML algorithm.
