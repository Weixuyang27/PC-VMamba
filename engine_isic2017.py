import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


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
    import os
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import torchvision.transforms as T
    from sklearn.metrics import confusion_matrix

    # 保存路径准备
    base_save = os.path.join(config.work_dir, 'outputs', 'val_pred')
    save_mask = os.path.join(base_save, 'masks')
    save_overlay_pred = os.path.join(base_save, 'overlay_pred')
    save_overlay_gt_pred = os.path.join(base_save, 'overlay_gt_pred')
    os.makedirs(save_mask, exist_ok=True)
    os.makedirs(save_overlay_pred, exist_ok=True)
    os.makedirs(save_overlay_gt_pred, exist_ok=True)

    # 模型评估模式
    model.eval()
    preds = []
    gts = []
    loss_list = []

    to_pil = T.ToPILImage()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())

            if type(out) is tuple:
                out = out[0]
            out_np = out.squeeze(1).cpu().detach().numpy()  # (B, H, W)
            preds.append(out_np)

            # 遍历 batch
            for b in range(out_np.shape[0]):
                pred = (out_np[b] >= config.threshold).astype(np.uint8) * 255
                gt = (msk[b, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                img_np = img[b].cpu()

                # 保存 mask
                mask_img = Image.fromarray(pred)
                mask_img.save(os.path.join(save_mask, f"mask_{idx * test_loader.batch_size + b}.png"))

                # 原图 (反归一化如有需要)
                if img_np.shape[0] == 1:
                    rgb_img = img_np.repeat(3, 0)
                else:
                    rgb_img = img_np
                rgb_img = to_pil(rgb_img)

                # overlay_pred (红色)
                overlay_pred = rgb_img.copy()
                pred_mask = Image.fromarray(pred).convert("L")
                red_mask = Image.new("RGB", pred_mask.size, (255, 0, 0))
                overlay_pred.paste(red_mask, mask=pred_mask)
                overlay_pred.save(
                    os.path.join(save_overlay_pred, f"overlay_pred_{idx * test_loader.batch_size + b}.png"))

                # overlay_gt_pred (绿 + 红)
                overlay_mix = rgb_img.copy()
                gt_mask = Image.fromarray(gt).convert("L")
                green_mask = Image.new("RGB", gt_mask.size, (0, 255, 0))
                overlay_mix.paste(green_mask, mask=gt_mask)
                overlay_mix.paste(red_mask, mask=pred_mask)
                overlay_mix.save(
                    os.path.join(save_overlay_gt_pred, f"overlay_gt_pred_{idx * test_loader.batch_size + b}.png"))

    # 评估指标
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

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
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
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

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

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)