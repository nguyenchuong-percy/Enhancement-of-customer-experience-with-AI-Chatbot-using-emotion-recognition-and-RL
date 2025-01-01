import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import numpy as np
from scipy.spatial import distance

from loss import mask_nll_loss
from seq2seq import *
from dataloading import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def convert_response(response, batch_size):
    size1 = len(response)
    size2 = batch_size
    np_res = np.zeros((size1, size2), dtype=np.int64)
    np_lengths = np.zeros(size2, dtype=np.int64)
    for i in range(size1):
        prov = response[i].cpu().numpy()
        for j in range(prov.size):
            np_lengths[j] = np_lengths[j] + 1
            if prov.size > 1:
                np_res[i][j] = prov[j]
            else:
                np_res[i][j] = prov
    res = torch.from_numpy(np_res)
    lengths = torch.from_numpy(np_lengths)
    return res, lengths


def convert_target(target, batch_size):
    size1 = len(target)
    size2 = batch_size
    np_res = np.zeros((size1, size2), dtype=np.int64)
    mask = np.zeros((size1, size2), dtype=np.bool_)
    np_lengths = np.zeros(size2, dtype=np.int64)
    for i in range(size1):
        prov = target[i].cpu().numpy()
        for j in range(prov.size):
            np_lengths[j] = np_lengths[j] + 1
            if prov.size > 1:
                np_res[i][j] = prov[j]
            else:
                np_res[i][j] = prov

            if np_res[i][j] > 0:
                mask[i][j] = True
            else:
                mask[i][j] = False

    res = torch.from_numpy(np_res)
    lengths = torch.from_numpy(np_lengths)
    mask = torch.from_numpy(mask)
    max_target_len = torch.max(lengths)  # .detach().numpy()
    return res, mask, max_target_len


def transform_tensor_to_same_shape_as(tensor, shape):
    size1, size2 = shape
    np_new_t = np.zeros((size1, size2), dtype=np.int64)
    np_new_mask = np.zeros((size1, size2), dtype=np.bool_)
    tensor_size1, tensor_size2 = tensor.size()
    for i in range(tensor_size1):
        for j in range(tensor_size2):
            np_new_t[i][j] = tensor[i][j]
            np_new_mask[i][j] = True
    return torch.from_numpy(np_new_t), torch.from_numpy(np_new_mask)


def calculate_bleu(references, hypotheses):
    """
    Calculate BLEU score for a set of references and hypotheses.

    Args:
        references (list of list of str): List of reference texts (each as a list of words).
        hypotheses (list of str): List of hypothesis texts (each as a single string).

    Returns:
        float: Average BLEU score for all pairs of references and hypotheses.
    """
    smooth = SmoothingFunction().method1
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = [ref.split()]
        hyp_tokens = hyp.split()
        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth)
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def calculate_rouge(references, hypotheses):
    """
    Calculate ROUGE scores for a set of references and hypotheses.

    Args:
        references (list of str): List of reference texts.
        hypotheses (list of str): List of hypothesis texts.

    Returns:
        dict: Average ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for all pairs of references and hypotheses.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    return {key: sum(scores) / len(scores) if scores else 0 for key, scores in rouge_scores.items()}

import os
import csv

def save_scores_to_csv(file_path, iteration, bleu_score, rouge_scores):
    # Check if file exists
    file_exists = os.path.isfile(file_path)
    
    # Open file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if the file doesn't exist
        if not file_exists:
            writer.writerow(["Iteration", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"])
        
        # Write the current scores
        writer.writerow([
            iteration,
            bleu_score,
            rouge_scores['rouge1'],
            rouge_scores['rouge2'],
            rouge_scores['rougeL']
        ])


