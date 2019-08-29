#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:49:45 2019

@author: stefan
"""
import torch


def BCEWithLogitsLoss(pred, target, weight=None):

    fn = torch.nn.BCEWithLogitsLoss()

    loss = fn(pred, target)
    
    return loss