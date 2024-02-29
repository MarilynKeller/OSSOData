ukbiobank_data_root = '/ps/data/SensitiveData/biobank/45117/20158-2.0/'
return_data_path = '/is/cluster/fast/mkeller2/Data/Skeleton/biobank_return/biobank_return/'

# DXA Landmarks predictors
joint_pred_skel_checkpoint = '/ps/project/rib_cage_breathing/Data/skeleton/checkpoints/Mar07_12-19-13_Input_skel_filtered_LR_0.00012_BS_12_ntrain_9000/CP_epoch3.pth'

nb_ldm = 29 #24
nstack = 8
jl_inp_dim = 256
jl_out_dim = 64
in_channels = 1
joint_pred_th = 0.95
