import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import numpy as np
from sympy import Symbol, solve
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, r2_score

import itertools


def writelog(file, line):
    file.write(line + "\n")
    print(line)

sequence_length = 11
def normalize_feature(train_data):
    '''
    Normalize to ICV
    Ventricles Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp
    :param dataset
    :return: normalized data
    '''
    tmp = []
    train_feature = train_data[:, :, 8:14]
    ICV_bl = train_data[:, :, 14]
    len = np.shape(train_feature)[-1]
    mask = np.ones_like(train_feature.reshape(-1,6))
    mask[np.where(train_feature.reshape(-1, 6) == 0)] = 0
    for idx in range(len):
        data = train_feature[:,:,idx]
        norm_data = np.true_divide(data, ICV_bl)
        tmp.append(norm_data)
        t_tmp = np.array(tmp).transpose(1, 2, 0)
    return t_tmp.astype(float), mask.reshape(-1,sequence_length ,6).astype(float)

def scaling_feature_t(train_feature, estim_m_out=None, estim_c_out=None, train=False):
    (b, s, f) = train_feature.shape
    tmp = train_feature.reshape(b*s, f)  # 26391 x 6
    norm_train_feature = []
    norm_estim_c = []
    norm_estim_m = []

    for idx in range(tmp.shape[1]):
        tmp_vol = tmp[:, idx]
        if train == True:
            tmp_vol_max = np.max(tmp)
            tmp_vol_min = np.min(tmp[np.nonzero(tmp)])
            m = Symbol('m')
            c = Symbol('c')
            equation1 = m * tmp_vol_max + c - 1
            equation2 = m * tmp_vol_min + c + 1
            estim_m = solve((equation1, equation2), dict=True)[0][m]
            estim_c = solve((equation1, equation2), dict=True)[0][c]
        else:
            estim_m = estim_m_out[idx]
            estim_c = estim_c_out[idx]
        norm_tmp_vol = (estim_m * tmp_vol) + estim_c
        norm_train_feature.append(norm_tmp_vol)
        norm_estim_m.append(estim_m)
        norm_estim_c.append(estim_c)
    norm_train_feature = np.array(norm_train_feature)
    norm_estim_m = np.array(norm_estim_m)
    norm_estim_c = np.array(norm_estim_c)

    norm_train_feature_t = norm_train_feature.transpose(1, 0).reshape(b, s, f)
    return norm_train_feature_t.astype(float), norm_estim_m.astype(float), norm_estim_c.astype(float)

def scaling_feature_e(train_feature, estim_m_out=None, estim_c_out=None, train=False):
    (b, s, f) = train_feature.shape
    tmp = train_feature.reshape(b*s, f)  # 26391 x 6
    norm_train_feature = []
    norm_estim_c = []
    norm_estim_m = []

    for idx in range(tmp.shape[1]):
        tmp_vol = tmp[:, idx]
        if train == True:
            tmp_vol_max = np.max(tmp[:, idx])
            tmp_vol_min = np.min(tmp[np.nonzero(tmp[:, idx]), idx])

            m = Symbol('m')
            c = Symbol('c')
            equation1 = m * tmp_vol_max + c - 1
            equation2 = m * tmp_vol_min + c + 1
            estim_m = solve((equation1, equation2), dict=True)[0][m]
            estim_c = solve((equation1, equation2), dict=True)[0][c]
        else:
            estim_m = estim_m_out[idx]
            estim_c = estim_c_out[idx]
        norm_tmp_vol = (estim_m * tmp_vol) + estim_c
        norm_train_feature.append(norm_tmp_vol)
        norm_estim_m.append(estim_m)
        norm_estim_c.append(estim_c)
    norm_train_feature = np.array(norm_train_feature)
    norm_estim_m = np.array(norm_estim_m)
    norm_estim_c = np.array(norm_estim_c)

    norm_train_feature_t = norm_train_feature.transpose(1, 0).reshape(b, s, f)
    return norm_train_feature_t.astype(float), norm_estim_m.astype(float), norm_estim_c.astype(float)

def masking_cogntive_score(data):
    tmp = []
    max_range = [30,70,85]
    cog_feature = data.copy()
    mask = np.ones_like(cog_feature.reshape(-1,3))
    mask[np.where(cog_feature.reshape(-1,3)==0)] = 0
    for i in range(cog_feature.shape[2]):
        cog_data = cog_feature[:,:,i]
        norm_data = cog_data / max_range[i]
        tmp.append(norm_data)
        t_tmp = np.array(tmp).transpose(1,2,0)
    return t_tmp.astype(float), mask.reshape(-1, sequence_length, 3).astype(int)


def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var



class BaseDataset(Dataset):
    def __init__(self, data, mask, label):
        self.data = torch.FloatTensor(data)
        self.mask = torch.FloatTensor(mask)
        self.label = torch.FloatTensor(label)
    def __getitem__(self, item):
        data = {'data':self.data[item], 'mask':self.mask[item], 'label':self.label[item]}
        return data

    def __len__(self):
        return len(self.data)

def sample_loader(data, mask, label, batch_size, is_train=True):
    # recs = {'label': label, 'data': data, 'mask': mask}
    # if is_train:
    #     recs['is_train'] = 1
    # else:
    #     recs['is_train'] = 0
    # loader = DataLoader(recs, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True,
    #                     collate_fn=collate_fn)
    loader = DataLoader(BaseDataset(data, mask, label), batch_size=batch_size, shuffle=False)
    return loader

def collate_fn(recs):
    rec_dict = {
        'data': torch.FloatTensor(np.array([r['data'] for r in recs])),
        'mask': torch.FloatTensor(np.array([r['mask'] for r in recs])),
        'label': torch.FloatTensor(np.array([r['label'] for r in recs])),
    }
    return rec_dict



def calculate_performance(y, y_score, y_pred, classes=None):
    # Calculate Evaluation Metrics
    acc = accuracy_score(y_pred, y) * 100
    if classes == None:
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    if classes == True:
        a, b, c, \
        d, e, f, \
        g, h, i = confusion_matrix(y, y_pred, labels=[0, 1, 2]).ravel()
        tn, fp, fn, tp = (a + c + g + i), (b + h), (d + f), (e)
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        if classes == None:
            p, r, t = precision_recall_curve(y, y_score)
            auprc = np.nan_to_num(metrics.auc(r, p))
        else:
            p, r, t = 0, 0, 0
            auprc = 0

    spec = np.nan_to_num(tn / (tn + fp))
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        if classes == None:
            auc = roc_auc_score(y, y_score)
        else:
            auc = 0
    except ValueError:
        auc = 0

    return auc, auprc, acc, balacc, sen, spec, prec, recall

def calculate_performance_ver2(y_label, y_pred, classes=None):
    if classes == None:
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred, labels=[0, 1]).ravel()

        return metrics.precision_score(y_label, y_pred, zero_division=1), metrics.recall_score(y_label, y_pred), (tn / (tn + fp)), (tn / (tn + fp)), ((tn / (tn + fp)) + (tn / (tn + fp))) / 2, None
    elif classes == True:
        a, b, c, \
        d, e, f, \
        g, h, i = confusion_matrix(y_label, y_pred, labels=[0, 1, 2]).ravel()

        precision = np.zeros(3)  # PPV: TP/(TP+FP)
        recall = np.zeros(3)  # TPR (sensitivity): TP/(TP+FN)
        specificity = np.zeros(3)  # TNR: TN/(TN+FP)

        tp_a = a
        fp_a = d + g
        fn_a = b + c
        tn_a = e + f + h + i
        epsilon = 1e-5
        precision[0] = tp_a / (tp_a + fp_a + epsilon)
        recall[0] = tp_a / (tp_a + fn_a + epsilon)
        specificity[0] = tn_a / (tn_a + fp_a + epsilon)

        tp_b = e
        fp_b = d + f
        fn_b = b + h
        tn_b = a + c + g + i
        precision[1] = tp_b / (tp_b + fp_b + epsilon)
        recall[1] = tp_b / (tp_b + fn_b  + epsilon)
        specificity[1] = tn_b / (tn_b + fp_b + epsilon)

        tp_c = i
        fp_c = g + h
        fn_c = c + f
        tn_c = a + b + d + e
        precision[2] = tp_c / (tp_c + fp_c + epsilon)
        recall[2] = tp_c / (tp_c + fn_c + epsilon)
        specificity[2] = tn_c / (tn_c + fp_c + epsilon)

        return np.mean(precision), np.mean(recall), np.mean(specificity), np.mean(recall), (
                np.mean(recall) + np.mean(specificity)) / 2, 2 * np.mean(precision) * np.mean(recall) / (
                       np.mean(precision) + np.mean(recall))
    else:
        print('Error!')

def a_value(probabilities, zero_label=0, one_label=1):
    """
    Approximates the AUC by the method described in Hand and Till 2001,
    equation 3.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        probabilities (list): A zipped list of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        zero_label (optional, int): The label to use as the class '0'.
            Must be an integer, see above for details.
        one_label (optional, int): The label to use as the class '1'.
            Must be an integer, see above for details.
    Returns:
        The A-value as a floating point.
    """
    # Obtain a list of the probabilities for the specified zero label class
    expanded_points = []
    for instance in probabilities:
        if instance[0] == zero_label or instance[0] == one_label:
            expanded_points.append((instance[0].item(), instance[zero_label+1].item()))
    sorted_ranks = sorted(expanded_points, key=lambda x: x[1])

    n0, n1, sum_ranks = 0, 0, 0
    # Iterate through ranks and increment counters for overall count and ranks of class 0
    for index, point in enumerate(sorted_ranks):
        if point[0] == zero_label:
            n0 += 1
            sum_ranks += index + 1  # Add 1 as ranks are one-based
        elif point[0] == one_label:
            n1 += 1
        else:
            pass  # Not interested in this class

    return (sum_ranks - (n0*(n0+1)/2.0)) / float(n0 * n1)  # Eqn 3

def MAUC(data, num_classes):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        data (list): A zipped list (NOT A GENERATOR) of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        num_classes (int): The number of classes in the dataset.
    Returns:
        The MAUC as a floating point value.
    """
    # Find all pairwise comparisons of labels
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]

    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(data, zero_label=pairing[0], one_label=pairing[1]) + a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0

    return sum_avals * (2 / float(num_classes * (num_classes-1)))  # Eqn 7


def regression_cog(imputations, evals, eval_masks, original_cog=None):
    x_hat = np.array(imputations).reshape(-1, 1)
    x_true = np.array(evals).reshape(-1, 1)
    masks = np.array(eval_masks).reshape(-1, 1)
    # original_cog = [30,70,85]
    y_hat = (x_hat[np.where(masks == 1)])
    y_true = (x_true[np.where(masks == 1)])

    mape = np.mean(np.abs((y_true - y_hat) / y_true))
    r_square = r2_score(y_true, y_hat)

    return mape, r_square


def calculate_summary(cache):
    k_fold = 5
    def get_item(cache, item):
        i = [c[item] for c in cache]
        return i
    acc = [round(np.mean(get_item(cache, 'Acc')), k_fold), round(np.std(get_item(cache, 'Acc')), k_fold)]
    auc = [round(np.mean(get_item(cache, 'MAUC')), k_fold), round(np.std(get_item(cache, 'MAUC')), k_fold)]
    sen = [round(np.mean(get_item(cache, 'Sens')), k_fold), round(np.std(get_item(cache, 'Sens')), k_fold)]
    spec = [round(np.mean(get_item(cache, 'Spec')), k_fold), round(np.std(get_item(cache, 'Spec')), k_fold)]
    prec = [round(np.mean(get_item(cache, 'Prec')), k_fold), round(np.std(get_item(cache, 'Prec')), k_fold)]
    recall = [round(np.mean(get_item(cache, 'Recall')), k_fold), round(np.std(get_item(cache, 'Recall')), k_fold)]
    return acc, auc, sen, spec, prec, recall