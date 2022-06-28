import torch
import torch.nn as nn
import torch.optim as optim

import os, argparse, datetime
import numpy as np
from tqdm import tqdm

from model import ODERGRU_imputation
from train import train_op, test_op
from loss import FocalLoss
from utils import *


def main(args):
    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1234)

    # Log
    directory = "./{}/".format(str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))) + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)

    f = open(directory + "_log.txt", 'a')
    writelog(f, "-" * 15)
    writelog(f, 'TRAINING PARAMETER')
    writelog(f, 'Dataset: ' + str(args.dataset))
    writelog(f, 'Learning Rate : ' + str(args.learning_rate))
    writelog(f, 'Weight Decay : ' + str(args.weight_decay))
    writelog(f, 'Batch Size : ' + str(args.batch_size))
    writelog(f, 'Latents Size : ' + str(args.latents))
    writelog(f, 'RGRU Hidden Size : ' + str(args.rgru_hid_size))
    writelog(f, 'Reg Weight: ' + str(args.reg_weight))
    writelog(f, 'Label Weight: ' + str(args.label_weight))
    writelog(f, 'gamma: ' + str(args.gamma))
    writelog(f, "-" * 15)
    writelog(f, 'TRAINING LOG')

    path = os.path.join(args.data_path, "Journal_Ghazi_{}_data_{}.npz")

    kfold=[]
    for k in range(args.kfold):
        ds = np.load(path.format(args.dataset, k), allow_pickle=True)
        train_X = ds['Train_data']
        train_y = ds['Train_label']
        valid_X = ds['Valid_data']
        valid_y = ds['Valid_label']
        test_X = ds['Test_data']
        test_y = ds['Test_label']

        # Normalization
        train_feature, train_mask = normalize_feature(train_X)
        valid_feature, valid_mask = normalize_feature(valid_X)
        test_feature, test_mask = normalize_feature(test_X)

        # Linearly normalize each of volumns
        if args.feature == 'total':
            norm_train_feature, estim_m, estim_c = scaling_feature_t(train_feature, None, None, train=True)
            norm_valid_feature, v_estim_m, v_estim_c = scaling_feature_t(valid_feature, estim_m, estim_c, train=False)
            norm_test_feature, t_estim_m, t_estim_c = scaling_feature_t(test_feature, estim_m, estim_c, train=False)
        else:
            norm_train_feature, estim_m, estim_c = scaling_feature_e(train_feature, None, None, train=True)
            norm_valid_feature, v_estim_m, v_estim_c = scaling_feature_e(valid_feature, estim_m, estim_c, train=False)
            norm_test_feature, t_estim_m, t_estim_c = scaling_feature_e(test_feature, estim_m, estim_c, train=False)

        ## Class case
        if args.binary_case == 1: #AD vs. MCI
            train_y[np.where(train_y == args.binary_case)] = -1
            valid_y[np.where(valid_y == args.binary_case)] = -1
            test_y[np.where(test_y == args.binary_case)] = -1

        if args.binary_case == 2: #AD vs. CN
            train_y[np.where(train_y == args.binary_case)] = -1
            valid_y[np.where(valid_y == args.binary_case)] = -1
            test_y[np.where(test_y == args.binary_case)] = -1

        if args.binary_case == 3: #MCI vs. CN
            train_y[np.where(train_y == args.binary_case)] = -1
            valid_y[np.where(valid_y == args.binary_case)] = -1
            test_y[np.where(test_y == args.binary_case)] = -1

        ## Cognitive score case
        if args.cognitive_score == True:
            mmse_train_feature = train_X[:, :, 3:6]
            mmse_valid_feature = valid_X[:, :, 3:6]
            mmse_test_feature = test_X[:, :, 3:6]

            train_cog_norm_feature, train_cog_norm_mask = masking_cogntive_score(mmse_train_feature)
            valid_cog_norm_feature, valid_cog_norm_mask = masking_cogntive_score(mmse_valid_feature)
            test_cog_norm_feature, test_cog_norm_mask = masking_cogntive_score(mmse_test_feature)

            model_train_input = np.concatenate((norm_train_feature, train_cog_norm_feature), axis=2)
            model_train_mask = np.concatenate((train_mask, train_cog_norm_mask), axis=2)
            model_valid_input = np.concatenate((norm_valid_feature, valid_cog_norm_feature), axis=2)
            model_valid_mask = np.concatenate((valid_mask, valid_cog_norm_mask), axis=2)
            model_test_input = np.concatenate((norm_test_feature, test_cog_norm_feature), axis=2)
            model_test_mask = np.concatenate((test_mask, test_cog_norm_mask), axis=2)

        train_y -= 1
        valid_y -= 1
        test_y -= 1

        # DataLoader
        if args.cognitive_score == True:
            train_loader = sample_loader(model_train_input, np.asarray(model_train_mask), train_y, args.batch_size,
                                         is_train=True)
            valid_loader = sample_loader(model_valid_input, np.asarray(model_valid_mask), valid_y,
                                         model_valid_input.shape[0])
            test_loader = sample_loader(model_test_input, np.asarray(model_test_mask), test_y,
                                        model_test_input.shape[0])
            dataloaders = {'train': train_loader,
                           'valid': valid_loader,
                           'test': test_loader}
        else:
            train_loader = sample_loader(norm_train_feature, np.asarray(train_mask), train_y, args.batch_size,
                                         is_train=True)
            valid_loader = sample_loader(norm_valid_feature, np.asarray(valid_mask), valid_y,
                                         norm_valid_feature.shape[0])
            test_loader = sample_loader(norm_test_feature, np.asarray(test_mask), test_y,
                                        norm_test_feature.shape[0])
            dataloaders = {'train': train_loader,
                           'valid': valid_loader,
                           'test': test_loader}

        # Define Model & Optimizer
        criterion_reg = nn.MSELoss()
        criterion_cls = FocalLoss(gamma = args.gamma, ignore_index=-2)
        criterion_imputation = [criterion_reg, criterion_cls]

        print("ODE-RGRU imputation model training...")
        model = ODERGRU_imputation(input_dim=9, latents=args.latents, rgru_hid_size=args.rgru_hid_size, n_layers=args.n_layers, ode_units=32, reg_weight=args.reg_weight, label_weight=args.label_weight).to(device) # ode units = 64
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        cache = {"Model": "ODE-RGRU", 'Fold': k, 'Epoch': 0,
                 "Acc": 0, "MAUC": 0, "Sens": 0,
                 'Spec': 0, 'Prec': 0, 'Recall': 0,
                 'Rho' : 0
                 }

        best_mauc = 0
        for epoch in range(args.epochs):
            train_op(model, dataloaders, optimizer, criterion_imputation)
            mauc, m_acc, m_sen, m_spec, m_prec, m_recall, data, ret = test_op(model, dataloaders, criterion_imputation, type='test')
            valid_mauc, _, _, _, _, _, _, _ = test_op(model, dataloaders, criterion_imputation, type='valid')

            plot_mape_cog1, plot_r2_cog1 = regression_cog(ret['predict_mmse'].data.cpu().numpy(),
                                                          data['data'][:, 1:, 6:7].data.cpu().numpy(),
                                                          data['mask'][:, 1:, 6:7].data.cpu().numpy(),
                                                          original_cog=np.array([30]))
            plot_mape_cog2, plot_r2_cog2 = regression_cog(ret['predict_ad11'].data.cpu().numpy(),
                                                          data['data'][:, 1:, 7:8].data.cpu().numpy(),
                                                          data['mask'][:, 1:, 7:8].data.cpu().numpy(),
                                                          original_cog=np.array([70]))
            plot_mape_cog3, plot_r2_cog3 = regression_cog(ret['predict_ad13'].data.cpu().numpy(),
                                                          data['data'][:, 1:, 8:9].data.cpu().numpy(),
                                                          data['mask'][:, 1:, 8:9].data.cpu().numpy(),
                                                          original_cog=np.array([85]))
            if valid_mauc > best_mauc:
                best_mauc = valid_mauc
                cache['Epoch'] = epoch
                cache['Acc'] = m_acc
                cache['MAUC'] = mauc
                cache['Sens'] = m_sen
                cache['Spec'] = m_spec
                cache['Prec'] = m_prec
                cache['Recall'] = m_recall

                cache['Cog1'] = plot_mape_cog1
                cache['Cog2'] = plot_mape_cog2
                cache['Cog3'] = plot_mape_cog3
                cache['Cog1_r2'] = plot_r2_cog1
                cache['Cog2_r2'] = plot_r2_cog2
                cache['Cog3_r2'] = plot_r2_cog3



                state = {"model" : model.state_dict()}
                torch.save(state, directory + "{}_ckpt.t7".format(k))
            print("TEST||{}---ACC:{:.4f}, AUC:{:.4f}, Sens:{:.4f}, Spec:{:.4f}, Prec:{:.4f}, Recall:{:.4f}".format(
                epoch, m_acc, mauc, m_sen, m_spec, m_prec, m_recall))
            print("{}/{}---ACC:{:.4f}, AUC:{:.4f}, Sens:{:.4f}, Spec:{:.4f}, Prec:{:.4f}, Recall:{:.4f}".format(
                cache['Epoch'], epoch, cache['Acc'], cache['MAUC'], cache['Sens'], cache['Spec'], cache['Prec'],
                cache['Recall']))

        writelog(f, '-' * 15)
        writelog(f, 'Summary of Fold :{}'.format(k))
        writelog(f, 'Best Epoch : {}'.format(cache['Epoch']))
        writelog(f, 'ACC : {:.4f}'.format(cache['Acc']))
        writelog(f, 'AUC : {:.4f}'.format(cache['MAUC']))
        writelog(f, 'Sens : {:.4f}'.format(cache['Sens']))
        writelog(f, 'Spec : {:.4f}'.format(cache['Spec']))
        writelog(f, 'Prec : {:.4f}'.format(cache['Prec']))
        writelog(f, 'Recall : {:.4f}'.format(cache['Recall']))

        writelog(f, 'Cog1 MAPE : ' + str(cache['Cog1']))
        writelog(f, 'Cog2 MAPE : ' + str(cache['Cog2']))
        writelog(f, 'Cog3 MAPE : ' + str(cache['Cog3']))

        writelog(f, 'Cog1 r2 : ' + str(cache['Cog1_r2']))
        writelog(f, 'Cog2 r2: ' + str(cache['Cog2_r2']))
        writelog(f, 'Cog3 r2: ' + str(cache['Cog3_r2']))

        kfold.append(cache)

    writelog(f, '-'*15)
    writelog(f, 'Summary of all KFOLD')
    acc, auc, sen, spec, prec, recall = calculate_summary(kfold)

    writelog(f, 'AUC : ' + str(auc[0]) + ' + ' + str(auc[1]))
    writelog(f, 'Accuracy : ' + str(acc[0]) + ' + ' + str(acc[1]))
    writelog(f, 'Sensitivity : ' + str(sen[0]) + ' + ' + str(sen[1]))
    writelog(f, 'Specificity : ' + str(spec[0]) + ' + ' + str(spec[1]))
    writelog(f, 'Precision : ' + str(prec[0]) + ' + ' + str(prec[1]))
    writelog(f, 'Recall : ' + str(recall[0]) + ' + ' + str(recall[1]))



    writelog(f, '---------------')
    writelog(f, 'END OF CROSS VALIDATION TRAINING')
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='AD progression') # TADPOLE Challenge
    parser.add_argument("-g", '--gpu', type=str, default="0")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Zero')
    parser.add_argument('--data_path', type=str, default='/DataRead/swjeong/TADPOLE/')
    parser.add_argument('--feature', type=str, default='total') # total or each
    parser.add_argument('--task', type=int, default=1)
    # parser.add_argument("--whichmodel", help="which model", type=str, default='model')

    parser.add_argument('--latents', type=int, default=32)
    parser.add_argument('--rgru_hid_size', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=1)

    parser.add_argument('--reg_weight', type=float, default=1.0)
    parser.add_argument('--label_weight', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=5.0)
    parser.add_argument('--binary_case', type=int, default=-1)
    parser.add_argument('--cognitive_score', type=bool, default=True)
    args = parser.parse_args()

    main(args)
