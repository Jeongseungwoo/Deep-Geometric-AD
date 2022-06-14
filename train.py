import numpy as np
from tqdm import tqdm

from utils import to_var, calculate_performance, calculate_performance_ver2, MAUC, regression

def train_op(model, ds, optimizer, criterion):
    model.train()
    bar = tqdm(ds['train'])
    loss = 0
    n = 0
    for i, data in enumerate(bar):
        data = to_var(data)
        ret = model.run_on_batch(data, optimizer, criterion[0], criterion[1], multi_flag=True)
        loss += ret['loss'].item()
        n += 1
        bar.set_description('Train||Loss:{:.4f} ({})'.format(loss/n, n))

def test_op(model, ds, criterion, type='test'):
    model.eval()
    bar = tqdm(ds[type])
    loss = 0
    n = 0
    for i, data in enumerate(bar):
        data = to_var(data)
        ret = model.run_on_batch(data, None, criterion[0], criterion[1], multi_flag=True)
        pred_score = ret['predict'].data.cpu().numpy()
        pred_label = np.argmax(ret['predict'].data.cpu().numpy(), axis=1)
        label = ret['labels'].data.cpu().numpy()

        loss += ret['loss'].item()
        n += 1

        filtered_pred_label = pred_label[np.where(label != -2)[0]]
        filtered_pred_score = pred_score[np.where(label != -2)[0]]
        filtered_label = label[np.where(label != -2)[0]]

        ## AD v.s CN
        label_y_AN = filtered_label[np.where(filtered_label != 1)[0]]
        label_y_AN[label_y_AN == 2] = 1
        pred_A_N = filtered_pred_score[np.where(filtered_label != 1)[0]][:, (0, 2)][:, 1:]
        pred_A_N_label = np.argmax(pred_A_N, 1)

        an_auc, an_auprc, an_acc, _, _, _, _, _ = calculate_performance(label_y_AN, pred_A_N, pred_A_N_label, None)
        an_prec, an_recall, an_spec, an_sen, an_balacc, _ = calculate_performance_ver2(label_y_AN, pred_A_N_label, None)

        ## AD v.s MCI
        label_y_AM = filtered_label[np.where(filtered_label != 0)[0]]
        label_y_AM[label_y_AM == 1] = 0
        label_y_AM[label_y_AM == 2] = 1
        pred_A_M = filtered_pred_score[np.where(filtered_label != 0)[0]][:, 1:][:, 1:]
        pred_A_M_label = np.argmax(pred_A_M, 1)

        am_auc, am_auprc, am_acc, _, _, _, _, _ = calculate_performance(label_y_AM, pred_A_M, pred_A_M_label, None)
        am_prec, am_recall, am_spec, am_sen, am_balacc, _ = calculate_performance_ver2(label_y_AM, pred_A_M_label, None)

        ## MCI v.s CN
        label_y_MN = filtered_label[np.where(filtered_label != 2)[0]]
        pred_M_N = filtered_pred_score[np.where(filtered_label != 2)[0]][:, 1:][:, 1:]
        pred_M_N_label = np.argmax(pred_M_N, 1)

        mn_auc, mn_auprc, mn_acc, _, _, _, _, _ = calculate_performance(label_y_MN, pred_M_N, pred_M_N_label, None)
        mn_prec, mn_recall, mn_spec, mn_sen, mn_balacc, _ = calculate_performance_ver2(label_y_MN, pred_M_N_label, None)

        ###
        mauc = MAUC(np.concatenate((filtered_label, filtered_pred_score), axis=1), num_classes=3)
        auc, m_auprc, m_acc, _, _, _, _, _ = calculate_performance(filtered_label, filtered_pred_score,
                                                                   filtered_pred_label, True)
        m_prec, m_recall, m_spec, m_sen, m_balacc, _ = calculate_performance_ver2(filtered_label, filtered_pred_label,
                                                                                  True)

        bar.set_description("{}||Loss:{:.4f} ({})".format(type, loss / n, n))

    return mauc, m_acc, m_sen, m_spec, m_prec, m_recall, data, ret