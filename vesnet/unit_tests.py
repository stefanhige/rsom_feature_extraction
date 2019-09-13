# tests for calc_metrics() in lossfunctions.py

import copy
import os
import torch
import numpy as np

import unittest
import math


# modules to get tested
from lossfunctions import calc_metrics
from dataloader import DataAugmentation



os.environ["CUDA_VISIBLE_DEVICES"]='0'

# helper functions
def _dice(x, y):
    '''
    do the test in numpy
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = x.astype(np.bool)
    y = y.astype(np.bool)

    i = np.logical_and(x,y)

    if x.sum() + y.sum() == 1:
        return 1.

    return (2. * i.sum()) / (x.sum() + y.sum())

def isclose(a, b):
    return math.isclose(a, b, rel_tol=1e-07)





# lossfunctions.calc_metrics

class TestCalcMetrics(unittest.TestCase):

    def test_dice_batch_size_1(self):

        # 2x2x2 volume
        # complete overlap
        pred = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        pred[0, 0, 0, 0, 0] = 1
        pred = pred.cuda()
        target = copy.deepcopy(pred)
        skel = copy.deepcopy(pred)

        metrics = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertEqual(dice, metrics['dice'])

        # two pixel vs one pixel
        target[0, 0, 0, 0, 1] = 1
        metrics = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertTrue(isclose(dice, metrics['dice']))
        
        # no overlap, zero
        target[0, 0, 0, 0, 0] = 0
        target[0, 0, 0, 0, 1] = 1
        metrics = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertEqual(dice, metrics['dice'])

    def test_dice_batch_size_2(self):

        # 2x2x2 volume
        # complete overlap
        pred = torch.zeros((2, 1, 2, 2, 2), dtype=torch.float32)
        pred[0, 0, 0, 0, 0] = 1
        pred = pred.cuda()
        target = copy.deepcopy(pred)
        skel = copy.deepcopy(pred)

        metrics = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertEqual(dice, metrics['dice'])

        # 1 pixel vs 3 pixel
        target[:, 0, 0, 0, 1] = 1
        metrics = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertTrue(isclose(dice, metrics['dice']))
        
        # no overlap, zero
        target[:, 0, 0, 0, :] = 0
        target[1, 0, 0, 0, 0] = 1
        metrics = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertEqual(dice, metrics['dice'])

    def test_cl_score_batch_size_1(self):
        # complete overlap, cl score = 1
        pred = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        pred[0, 0, 0, 0, 0] = 1
        pred = pred.cuda()
        target = copy.deepcopy(pred)
        skel = copy.deepcopy(pred)

        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(1., metrics['cl_score'])

        #skeleton enclosed
        pred[0, 0, :, :, :] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(1., metrics['cl_score'])

        # skeleton outside
        pred[:] = 0
        pred[0, 0, 0, 0, 1] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(0., metrics['cl_score'])

        # pixel enclosed 2 pixel outside
        # cl_score 0.333
        skel[0, 0, 0, 0, :] = 1
        skel[0, 0, 0, 1, 0] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertTrue(isclose(1/3, metrics['cl_score']))

    def test_cl_score_batch_size_2(self):
         # complete overlap, cl score = 1
        pred = torch.zeros((2, 1, 2, 2, 2), dtype=torch.float32)
        pred[0, 0, 0, 0, 0] = 1
        pred = pred.cuda()
        target = copy.deepcopy(pred)
        skel = copy.deepcopy(pred)
        
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(1., metrics['cl_score'])

        #skeleton enclosed
        pred[0, 0, :, :, :] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(1., metrics['cl_score'])

        # skeleton outside
        pred[:] = 0
        pred[:, 0, 0, 0, 1] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(0., metrics['cl_score'])

        # 2 pixel enclosed 6 pixel outside
        # cl_score 1/3
        skel[:, 0, 0, 0, :] = 1
        skel[:, 0, 0, 1, 0] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertTrue(isclose(1/3, metrics['cl_score']))

    def test_out_score_batch_size_1(self):
        
        # hull encludes everything but not corner elements, eg [0,0,0,0,0]
        target = torch.zeros((1, 1, 7, 7, 7), dtype=torch.float32)
        target[0, 0, 3, 3, 3] = 1
        target = target.cuda()
        skel = copy.deepcopy(target)
        
        # same, outside score = 1
        pred = copy.deepcopy(target)
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(1., metrics['out_score'])
    
        # pred completely outside 
        pred = torch.zeros((1, 1, 7, 7, 7), dtype=torch.float32)
        pred = pred.cuda()
        pred[0, 0, 0, 0, 0] = 1
        pred[0, 0, 0, 0, 6] = 1
        pred[0, 0, 0, 6, 0] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(0., metrics['out_score'])

        # pred partially inside
        pred = torch.zeros((1, 1, 7, 7, 7), dtype=torch.float32)
        pred = pred.cuda()
        pred[0, 0, 0, 0, 5:7] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(0.5, metrics['out_score'])

    def test_out_score_batch_size_2(self):
        # hull encludes everything but not corner elements, eg [0,0,0,0,0]
        target = torch.zeros((2, 1, 7, 7, 7), dtype=torch.float32)
        target[0, 0, 3, 3, 3] = 1
        target = target.cuda()
        skel = copy.deepcopy(target)
        
        # same, outside score = 1
        pred = copy.deepcopy(target)
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(1., metrics['out_score'])
    
        # pred completely outside 
        pred = torch.zeros((2, 1, 7, 7, 7), dtype=torch.float32)
        pred = pred.cuda()
        pred[0, 0, 0, 0, 0] = 1
        pred[0, 0, 0, 0, 6] = 1
        pred[0, 0, 0, 6, 0] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(0., metrics['out_score'])

        # pred partially inside
        pred = torch.zeros((2, 1, 7, 7, 7), dtype=torch.float32)
        pred = pred.cuda()
        pred[0, 0, 0, 0, 5:7] = 1
        metrics = calc_metrics(pred, target, skel)
        self.assertEqual(0.5, metrics['out_score'])

# dataloader.DataAugmentation

class TestDataloaderDataAugmentation(unittest.TestCase):
    def test_rescale_and_dims(self):
        data = np.random.randint(0, 255, size=(10, 9, 8, 1))
        label = np.random.randint(0, 255, size=(10, 9, 8, 1))
        meta = {'filename': 'R_20190101180101'}
        sample = {'data': data, 'label': label, 'meta': meta}

        tf = DataAugmentation()
        out = tf(sample)
        # print(out['data'].shape)

        self.assertTrue(np.amax(out['data']) <= 255)
        self.assertTrue(np.amax(out['label']) <= 255)
        self.assertTrue(np.amin(out['data']) >= 0)
        self.assertTrue(np.amin(out['label']) >= 0)
        self.assertEqual(np.prod(data.shape), np.prod(out['data'].shape))
        self.assertEqual(np.prod(label.shape), np.prod(out['label'].shape))
        self.assertTrue(out['data'].shape[-1] == 1)
        self.assertTrue(out['label'].shape[-1] == 1)

    def test_not_rom(self):
        data = np.random.randint(0, 255, size=(10, 9, 8, 1))
        label = np.random.randint(0, 255, size=(10, 9, 8, 1))
        meta = {'filename': '25.nii.gz'}
        sample = {'data': data, 'label': label, 'meta': meta}

        tf = DataAugmentation()
        out = tf(sample)
        self.assertTrue(np.all(data == out['data']))
        self.assertTrue(np.all(label == out['label']))

suiteList = []
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestCalcMetrics))
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestDataloaderDataAugmentation))

suite = unittest.TestSuite(suiteList)
unittest.TextTestRunner(verbosity=2).run(suite)



















