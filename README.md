# 1、What is DeepATPseq?
DeepATPseq is a predictor for the prediction of protein-ATP binding residues.
  

# 2、How to use the DeepATPseq? 
(1) Download all files and folders

(2) java -jar FileUnion.jar folder_path save_file_path 

Here, folder_path = ./DeepATPseq/DCNN/  save_file_path = ./DeepATPseq/DCNN.model and folder_path = ./DeepATPseq/SVM/  save_file_path = ./DeepATPseq/SVM.model

(3) Configure config.properties with the README

dcnn_model=./DeepATPseq/DCNN.model		            #represents the prediction model of DCNN
svm_model=./DeepATPseq/SVM.model		              #represents the prediction model of SVM
DeepATPseq_model=./DeepATPseq/DeepATPseq.py	      #represents the run file for the DeepATPSeq framework
dcnn_test_file=./DeepATPseq/Testdcnn.py		        #represents the run file for the DCNN framewor
python_environment=python	                        #represents the path to your native Python runtime environment
HHBLITS_DB_PATH=./uniclust30_2018_0               #HHBLITS_DB_PATH represents the path to the unicLust database

(4) java -jar deepatpseq.jar [input]Protein_name(String) [input]Protein_sequence(String) [input]Folder path to save the relevant files

After running step (4), the prediction results of the protein-ATP binding residues of the predicted proteins were obtained, and the result file was located in the following path: "Folder path to save the relevant files "/seq.deepatpseq.


# 3、Reference
Jun Hu, Lin-Lin Zheng, Yan-Song Bai, Ke-Wen Zhang, Dong-Jun Yu, Gui-Jun Zhang,
Sequence-based prediction of protein-ATP binding residues using deep convolutional neural network and support vector machine.

       	




