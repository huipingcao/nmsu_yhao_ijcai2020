# nmsu_yhao_ijcai2020
This is the GitHub repository for our publication "A new attention mechanism to classify multivariate time series", by Yifan Hao and Huiping Cao, which has been accepted to be published in IJCAI 2020. 

This paper uses 14 datasets. Each dataset contains two files: train.txt and test.txt.

For the 7 smaller datasets (file size is less than the GitHub limit, 100M), we directly put the processed data of these datasets in the data folder. 
For the 7 larger datasets (whose size is larger than the GitHub limit, 100M), we put the processed data of these datasets at google drive. These files can be downloaded from: https://drive.google.com/open?id=1eMLQIjDCvMIBs_BHDZAmb37MnlywxgyT

1. Prerequirments:
    The project is writen using Python 2.7. 
    The following packages are required to run this project:
    1.1 tensorflow-gpu-1.15.0
    1.2 scikit-learn 0.22.0
    1.3 numpy 1.17.3

2. Generate results in Table 2 and Figure 3
    2.1 Script:
        # python fcn_ca_main.py <DATA_NAME> <ATTENTION_TYPE>

    2.2 Parameters:
        <DATA_NAME>: dataset name. Not case sensitive. 
        14 possible values: act, atn, ara, aus, dsp, eeg, eeg2, ges, har, hts, jvo, net, ohc, ozo.
        This parameter is also used to identify the parameter file.
        For example, for "ges" dataset, the parameter file is 
        parameters/all_feature_classification_ges.txt. 

        <ATTENTION_TYPE>: the method we can test. 
        4 possible values: -1, 0, 1, 2.
        -1 means SFCN (Stablized Fully-Convolutional Network without any attention)
        0 means CA-SFCN (Cross-Attention Stablized Fully-Convolutional Network)
        1 means GA-SFCN (Global-Attention Stablized Fully-Convolutional Network)
        2 means RA-SFCN (Recurrent-Attention Stablized Fully-Convolutional Network)

        For example, the command "python fcn_ca_main.py ges 0"
        runs CA-SFCN method on ges Dataset.

    2.3 Outputs:
        The log file of the training stage locates at log/<DATASET_NAME>/fcn_classification/
        For example, the command "python fcn_ca_main.py ges 0"
        The output log file is
            log/ges/fcn_classification/ges_train_fcn_classification_act3_acc_attention0_conv3.log_<TIME_STAMP>.log

        The testing accuracy, training time, and testing time can be found at the last three rows of the output log file.

        The testing accuracies are reported in the last four columns of Table 2 in the paper. 
        The data in the other columns are directly copied from [Karim et al., 2019].
        The running time information is reported in Figure 3. 


3. Generate results in Table 3
    3.1 The results in the previous section are generated using SFCN model with 3 convolutional layers. 
        To build a SFCN model with only 1 convolutioanl layer, rename the parameter cnn_model_parameter_conv1.txt to be 
        cnn_model_parameter.txt.
        cnn_model_parameter_conv1.txt is the cnn setting parameter file with only one convolutional layer
    3.2 Re-run the script in 2.1
    3.3 The accuracy, running time results can be found in the log file: 
        For example, the command "python fcn_ca_main.py ges 0"
        The output log file is
        log/ges/fcn_classification/eeg_train_0_fcn_classification_act3_acc_attention0_conv1.log_<TIME_STAMP>.log
    3.4 Similar setting for the cnn with 5 convolutional layers using cnn_model_parameter_conv5.txt

    The testing accuracies in this section are reported in Table 3.

