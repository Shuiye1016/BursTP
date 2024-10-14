import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import numpy as np
import pickle as pkl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from argparse import ArgumentParser
from model.MyDataset import get_batch_data
from model.BursTP import BursTP
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

pl.seed_everything(0, workers=True)

def train():
    # parser
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BursTP.setting_model_args(parser)
    args = parser.parse_args()

    train_data = get_batch_data('./dataset/{}/train.pkl'.format(args.data_name), batch_size=args.batch_size,
                                seq_len=args.seq_len, label_len=args.label_len, type='train')
    valid_data = get_batch_data('./dataset/{}/val.pkl'.format(args.data_name), batch_size=args.batch_size,
                                seq_len=args.seq_len, label_len=args.label_len, type='valid')
    test_data = get_batch_data('./dataset/{}/test.pkl'.format(args.data_name), batch_size=args.batch_size,
                               seq_len=args.seq_len, label_len=args.label_len, type='test')

    with open('./dataset/{}/nodes.pkl'.format(args.data_name), 'rb') as f:
        nodes = pkl.load(f)
        N = len(nodes)
        args.N = N + 1

    model = BursTP(args.d_model, args.hidden_dim, args.gcn_layers, args.N, args.dropout_rate,
                          args.seq_len, args.nhead, args.ffn_dim, args.act, args.num_layers,
                          args.classes, args.label_len, args.lr, args.weight_decay, args.time_loss_weight)
    pl.seed_everything()

    checkpoints_callback = ModelCheckpoint(monitor='val_loss', filename=args.data_name + '-{epoch:03d}-{val_loss:.5f}',
                                           save_top_k=1, mode='min', save_last=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')

    trainer = pl.Trainer(
        callbacks=[checkpoints_callback, LearningRateMonitor(logging_interval='epoch'), early_stopping],
        max_epochs=500, gradient_clip_val=args.clip_val)

    trainer.tune(model=model, train_dataloaders=train_data, val_dataloaders=valid_data)
    trainer.fit(model=model, train_dataloaders=train_data, val_dataloaders=valid_data)
    res = trainer.test(model=model, dataloaders=test_data, ckpt_path='best')
    print(res)

if __name__ == '__main__':
    startTime = time.time()
    train()
    endTime = time.time()
    useTime = (endTime - startTime) / 60
    print("training %s" % useTime)