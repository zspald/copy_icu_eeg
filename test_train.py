from train import EEGLearner

# Train a CNN model on sample patient data
# patient_list = ["ICUDataRedux_0061", "ICUDataRedux_0062", "ICUDataRedux_0063", "ICUDataRedux_0064", "ICUDataRedux_0065", 
#                 "ICUDataRedux_0066", "ICUDataRedux_0068", "ICUDataRedux_0069"]
# patient_list = ["CNT684", "CNT685", "CNT687", "CNT688", "CNT689", "CNT690", "CNT691", "CNT692", "CNT694", "CNT695", 
#                 "CNT698", "CNT700", "CNT701", "CNT702", "CNT705", "CNT706", "ICUDataRedux_0054", "ICUDataRedux_0061", 
#                 "ICUDataRedux_0062", "ICUDataRedux_0063", "ICUDataRedux_0064", "ICUDataRedux_0065", "ICUDataRedux_0066", 
#                 "ICUDataRedux_0068", "ICUDataRedux_0069", "ICUDataRedux_0072", "ICUDataRedux_0073", "ICUDataRedux_0074", 
#                 "ICUDataRedux_0078", "ICUDataRedux_0082", "ICUDataRedux_0083", "ICUDataRedux_0084", "ICUDataRedux_0085", 
#                 "ICUDataRedux_0086", "ICUDataRedux_0087", "ICUDataRedux_0089", "ICUDataRedux_0090", "ICUDataRedux_0091"]
patient_list_cv_false = ["CNT684", "CNT687", "CNT689", "CNT690", "CNT691", "CNT692", "CNT694", "CNT695",
                         "CNT698", "CNT700", "CNT701", "CNT702", "CNT705", "CNT706", "ICUDataRedux_0054", "ICUDataRedux_0061",
                         "ICUDataRedux_0063", "ICUDataRedux_0064", "ICUDataRedux_0065", "ICUDataRedux_0066",
                         "ICUDataRedux_0068", "ICUDataRedux_0069", "ICUDataRedux_0072", "ICUDataRedux_0073", "ICUDataRedux_0074",
                         "ICUDataRedux_0078", "ICUDataRedux_0082", "ICUDataRedux_0083", "ICUDataRedux_0084",
                         "ICUDataRedux_0086", "ICUDataRedux_0087", "ICUDataRedux_0089", "ICUDataRedux_0090", "ICUDataRedux_0091"]

print("========== Training example ==========")
# train_module = EEGLearner(patient_list_cv_true)
train_module = EEGLearner(patient_list_cv_false)
# print(len(train_module.patient_list))
train_module.train_cnn(epochs=50, control=1800, cross_val=False, save=False, verbose=1)
# train_module.train_convolutional_gru(epochs=20, batch_size=100, control=1800, save=True, seq_len=20, verbose=1)
# train_module.train_conv_lstm(epochs=20, cross_val=False, save=True, seq_len=20, verbose=1)
