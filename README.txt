##DeepATPSeq is a predictor for the prediction of protein-ATP binding residues.

#####Step 1: Unzip jar.zip
#####Step 2: Configure config.properties with the README
#####Step 3: java -jar DeepATPSeq.jar [input]Protein_name(String) [input]Protein_sequence(String) [input]Folder path to save the relevant files


After running step 3, the prediction results of the protein-ATP binding residues of the predicted proteins were obtained, 
and the result file was located in the following path: "Folder path to save the relevant files "/seq.deepatpseq.
       	

/DeepATPseq/DCNN.model		#represents the prediction model of DCNN
/DeepATPseq/SVM.model		#represents the prediction model of SVM
/DeepATPseq/DeepATPseq.py	#represents the run file for the DeepATPSeq framework
/DeepATPseq/Testdcnn.py		#represents the run file for the DCNN framewor
python_environment=python	#represents the path to your native Python runtime environment


##HHBLITS_DB_PATH represents the path to the unicLust database
HHBLITS_DB_PATH=/data/commonuser/library/uniclust30_2018_08/uniclust30_2018_08

