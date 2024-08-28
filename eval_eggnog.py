import os.path

import click as ck
import pandas as pd
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
import math
from deepgo.torch_utils import FastTensorDataLoader
from deepgo.models import MLPModel
from deepgo.data import load_data
from deepgo.utils import Ontology, propagate_annots
from multiprocessing import Pool
from functools import partial
from deepgo.metrics import compute_roc


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='cc', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--model-name', '-m', type=ck.Choice([
        'eggnog']),
    default='eggnog',
    help='Prediction model name')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
@ck.option(
    '--egg_pred_root', '-epr', default="/home/share/huadjyin/home/wangchuyao1/gllm/data/GO/",
    help='eggnog result path')
def main(data_root, ont, model_name, test_data_name, egg_pred_root):
    go_file = f'{data_root}/go.obo'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}.pkl'

    go = Ontology(go_file, with_rels=True)

    test_data_file = 'test_data.pkl'
    test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_file}')

    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    # Loading best model
    print('Loading the best model')
    pred_path = f"{egg_pred_root}/{ont}_eggnog_mapper_predict.pkl"
    predict_df = pd.read_pickle(pred_path)
    predicts = merge_df(test_df, predict_df, terms_dict)
    # Propagate scores using ontology structure
    with Pool(32) as p:
        preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), list(predicts))

    test_df['preds'] = preds

    test_df.to_pickle(out_file)


def merge_df(test_df, pred_df, terms_dict) -> np.ndarray:
    """
    merge pred into test_df, url: https://alidocs.dingtalk.com/i/nodes/o14dA3GK8g5yoxRmHgv03xoZV9ekBD76?utm_scene=team_space
    :param test_df:
    :param pred_df: columns: index, protein, eggnog-mapper:{"predict":[]}
    :param terms_dict:
    :return:  ndarrary, shape: (len(test_df), len(terms_dict))
    """
    predict_egg_label = np.zeros((len(test_df), len(terms_dict)), dtype=np.int32)
    pred_df = pred_df.drop(columns=['protein'])
    merge_df = pd.merge(test_df, pred_df, on='index', how="left")
    terms_no_exit = []
    no_predict = []

    for i, predict in enumerate(merge_df["eggnog-mapper"]):
        if isinstance(predict, dict) and predict.get("predict", []) is not None:
            no_exit = []
            for name in predict.get("predict", []):
                if name in terms_dict:
                    predict_egg_label[i, terms_dict[name]] = 1
                else:
                    no_exit.append(name)
            if len(no_exit) > 0:
                print(f"{i}: no exit terms: {no_exit}")
                terms_no_exit.extend(no_exit)
        else:
            print(f"no predict for {i}")
            no_predict.append(i)
    terms_no_exit = list(set(terms_no_exit))

    print(f"no predict nums: {len(no_predict)}. terms: {no_predict[:10]}")
    print(f"no exit nums: {len(terms_no_exit)}. terms: {terms_no_exit[:10]}")
    return predict_egg_label


if __name__ == '__main__':
    main()
