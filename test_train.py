from train import EEGLearner
import numpy as np

# Train a CNN model on sample patient data
# patient_list = ["ICUDataRedux_0061", "ICUDataRedux_0062", "ICUDataRedux_0063", "ICUDataRedux_0064", "ICUDataRedux_0065", 
#                 "ICUDataRedux_0066", "ICUDataRedux_0068", "ICUDataRedux_0069"]
# patient_list = ["CNT684", "CNT685", "CNT687", "CNT688", "CNT689", "CNT690", "CNT691", "CNT692", "CNT694", "CNT695", 
#                 "CNT698", "CNT700", "CNT701", "CNT702", "CNT705", "CNT706", "ICUDataRedux_0054", "ICUDataRedux_0061", 
#                 "ICUDataRedux_0062", "ICUDataRedux_0063", "ICUDataRedux_0064", "ICUDataRedux_0065", "ICUDataRedux_0066", 
#                 "ICUDataRedux_0068", "ICUDataRedux_0069", "ICUDataRedux_0072", "ICUDataRedux_0073", "ICUDataRedux_0074", 
#                 "ICUDataRedux_0078", "ICUDataRedux_0082", "ICUDataRedux_0083", "ICUDataRedux_0084", "ICUDataRedux_0085", 
#                 "ICUDataRedux_0086", "ICUDataRedux_0087", "ICUDataRedux_0089", "ICUDataRedux_0090", "ICUDataRedux_0091"]
# patient_list_cv_false = ["CNT684", "CNT687", "CNT689", "CNT690", "CNT691", "CNT692", "CNT694", "CNT695",
#                          "CNT698", "CNT700", "CNT701", "CNT702", "CNT705", "CNT706", "ICUDataRedux_0054", "ICUDataRedux_0061",
#                          "ICUDataRedux_0063", "ICUDataRedux_0064", "ICUDataRedux_0065", "ICUDataRedux_0066",
#                          "ICUDataRedux_0068", "ICUDataRedux_0069", "ICUDataRedux_0072", "ICUDataRedux_0073", "ICUDataRedux_0074",
#                          "ICUDataRedux_0078", "ICUDataRedux_0082", "ICUDataRedux_0083", "ICUDataRedux_0084",
#                          "ICUDataRedux_0086", "ICUDataRedux_0087", "ICUDataRedux_0089", "ICUDataRedux_0090", "ICUDataRedux_0091"]

pt_list_nsz = np.array([
                        # 'CNT684',
                        'CNT685', 'CNT687', 'CNT688', 'CNT689', 'CNT690', 'CNT691', 
                        'CNT692', 'CNT694', 'CNT695', 'CNT698', 'CNT700', 'CNT701', 'CNT702', 
                        'CNT705', 'CNT706', 'CNT708', 'CNT710', 'CNT711', 'CNT713', 
                        # 'CNT715', 
                        'CNT720', 
                        # 'CNT723', 
                        'CNT724', 'CNT725', 
                        # 'CNT726', 
                        'CNT729', 'CNT730', 
                        'CNT731', 'CNT732', 'CNT733', 'CNT734', 'CNT737', 'CNT740', 'CNT741', 
                        'CNT742', 'CNT743', 
                        # 'CNT748', 
                        'CNT750', 'CNT757', 'CNT758', 'CNT765', 
                        'CNT773', 'CNT774', 'CNT775', 'CNT776', 'CNT778', 'CNT782', 
                        'ICUDataRedux_0023', 'ICUDataRedux_0026', 'ICUDataRedux_0029',
                        'ICUDataRedux_0030', 'ICUDataRedux_0034', 'ICUDataRedux_0035', 
                        'ICUDataRedux_0043', 'ICUDataRedux_0044', 'ICUDataRedux_0047', 
                        'ICUDataRedux_0048'
                        ])

pt_list_sz = np.array([
                       'ICUDataRedux_0060', 'ICUDataRedux_0061', 'ICUDataRedux_0062',
                       'ICUDataRedux_0063', 'ICUDataRedux_0064', 'ICUDataRedux_0065',
                       'ICUDataRedux_0066', 'ICUDataRedux_0067', 'ICUDataRedux_0068',
                       'ICUDataRedux_0069', 'ICUDataRedux_0072', 'ICUDataRedux_0073',
                       'ICUDataRedux_0074', 'ICUDataRedux_0054', 'ICUDataRedux_0078',
                       'ICUDataRedux_0082', 'ICUDataRedux_0083', 'ICUDataRedux_0084',
                       'ICUDataRedux_0085', 'ICUDataRedux_0086', 'ICUDataRedux_0087',
                       'ICUDataRedux_0089', 'ICUDataRedux_0090', 'ICUDataRedux_0091',
                       'ICUDataRedux_0003', 'ICUDataRedux_0004', 'ICUDataRedux_0006',
                       'CNT929', 'ICUDataRedux_0027', 'ICUDataRedux_0028', 'ICUDataRedux_0033',
                       'ICUDataRedux_0036', 'ICUDataRedux_0040', 'ICUDataRedux_0042',
                       'ICUDataRedux_0045', 'ICUDataRedux_0049', 'ICUDataRedux_0050'
                       ])

pt_list = np.r_[pt_list_nsz, pt_list_sz]

print("========== Training example ==========")
# train_module = EEGLearner(patient_list_cv_true)
train_module = EEGLearner(pt_list, pt_list_sz, pt_list_nsz)
# print(len(train_module.patient_list))
train_module.train_cnn(epochs=50, control=1800, kfold=True, save=True, verbose=1)
# train_module.train_convolutional_gru(epochs=20, batch_size=100, control=1800, save=True, seq_len=20, verbose=1)
# train_module.train_conv_lstm(epochs=20, cross_val=False, save=True, seq_len=20, verbose=1)
