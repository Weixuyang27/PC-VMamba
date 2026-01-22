import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs

try:
    from medpy.metric import binary
except ImportError:
    binary = None
    print("Warning: 'medpy' library not found. HD95 metric will not be calculated.")
    print("Please install it using: pip install medpy")

def calculate_mcc(TP, TN, FP, FN):
    """
    计算 Matthews Correlation Coefficient (MCC)
    """
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denominator == 0:
        return 0
    return numerator / denominator

def calculate_hd95_per_batch(preds_batch, gts_batch):
    """
    计算一个 batch 内的平均 HD95
    preds_batch: numpy array, 0/1 binary
    gts_batch: numpy array, 0/1 binary
    """
    if binary is None:
        return 0
    
    batch_hd95 = []
    # 确保是 (Batch, H, W) 形式
    if len(preds_batch.shape) == 2:
        preds_batch = preds_batch[np.newaxis, ...]
        gts_batch = gts_batch[np.newaxis, ...]
        
    for i in range(preds_batch.shape[0]):
        pred = preds_batch[i]
        gt = gts_batch[i]
        
        # HD95 要求预测和真值均不为空，否则无法计算距离
        if np.sum(pred) > 0 and np.sum(gt) > 0:
            try:
                res = binary.hd95(pred, gt)
                batch_hd95.append(res)
            except:
                continue
        elif np.sum(pred) == 0 and np.sum(gt) == 0:
            # 如果两者都为空，距离为0
            batch_hd95.append(0)
        else:
            continue
            
    if len(batch_hd95) > 0:
        return np.mean(batch_hd95)
    else:
        return np.nan

def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    hd95_list = [] # 新增：用于存储每个 batch 的 HD95

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            
            # 准备数据用于 Metric 计算
            np_gts = msk.squeeze(1).cpu().detach().numpy()
            gts.append(np_gts)
            
            if type(out) is tuple:
                out = out[0]
            np_out = out.squeeze(1).cpu().detach().numpy()
            preds.append(np_out)

            # 必须在 flatten 之前计算
            batch_pred_mask = np.where(np_out >= config.threshold, 1, 0)
            batch_gt_mask = np.where(np_gts >= 0.5, 1, 0)
            
            hd95_val = calculate_hd95_per_batch(batch_pred_mask, batch_gt_mask)
            if not np.isnan(hd95_val):
                hd95_list.append(hd95_val)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        
        # --- 新增: 计算 MCC 和 平均 HD95 ---
        mcc = calculate_mcc(TP, TN, FP, FN)
        avg_hd95 = np.mean(hd95_list) if len(hd95_list) > 0 else 0
  
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, \
                accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}, \
                mcc: {mcc:.4f}, hd95: {avg_hd95:.4f}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list), miou


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    hd95_list = [] # 新增：用于存储每个 batch 的 HD95

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            
            np_msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(np_msk)
            
            if type(out) is tuple:
                out = out[0]
            np_out = out.squeeze(1).cpu().detach().numpy()
            preds.append(np_out)
            
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

            batch_pred_mask = np.where(np_out >= config.threshold, 1, 0)
            batch_gt_mask = np.where(np_msk >= 0.5, 1, 0)
            
            hd95_val = calculate_hd95_per_batch(batch_pred_mask, batch_gt_mask)
            if not np.isnan(hd95_val):
                hd95_list.append(hd95_val)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        
        # 新增: 计算 MCC 和 平均 HD95 ---
        mcc = calculate_mcc(TP, TN, FP, FN)
        avg_hd95 = np.mean(hd95_list) if len(hd95_list) > 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
            
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, \
                accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}, \
                mcc: {mcc:.4f}, hd95: {avg_hd95:.4f}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
