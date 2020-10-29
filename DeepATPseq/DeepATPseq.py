import sys


def HandleFile():
    f_svm = open(svm_prod, 'r')
    f_dnn = open(dcnn_prod, 'r')

    label = []
    svm_bind = []
    svm_nobind = []
    dnn_bind = []
    dnn_nobind = []

    for eachline in f_svm.readlines():
        lines_svm = eachline.split('\t')
        label.append(lines_svm[0])
        svm_bind.append(lines_svm[1])
        svm_nobind.append(lines_svm[2])

    for eachline in f_dnn.readlines():
        lines_dnn = eachline.split('\t')
        dnn_bind.append(lines_dnn[1])
        dnn_nobind.append(lines_dnn[2])

    f_svm.close()
    f_dnn.close()

    f_prod = open(deepatpseq_prod, 'w')
    for i in range(len(label)):
        prod_bind = w * float(svm_bind[i]) + (1 - w) * float(dnn_bind[i])
        prod_nobind = w * float(svm_nobind[i]) + (1 - w) * float(dnn_nobind[i])
        prod_bind = format(float(prod_bind), '.6f')
        prod_nobind = format(float(prod_nobind), '.6f')
        if float(prod_bind) > 0.2:
            f_prod.writelines('1.000000' + '\t' + str(prod_bind) + '\t' + str(prod_nobind) + '\n')
        else:
            f_prod.writelines('0.000000' + '\t' + str(prod_bind) + '\t' + str(prod_nobind) + '\n')
    f_prod.close()


if __name__ == '__main__':
    w = 0.75
    dcnn_prod = sys.argv[1]
    svm_prod = sys.argv[2]
    deepatpseq_prod = sys.argv[3]
    # dnn_file = 'C:/Users/ZLinlin/Desktop/DeepATPseq/dcnn.prod'
    # svm_file = 'C:/Users/ZLinlin/Desktop//DeepATPseq/svm.prod'
    # file = 'C:/Users/ZLinlin/Desktop/DeepATPseq/deepatpseq.prod'
    HandleFile()
