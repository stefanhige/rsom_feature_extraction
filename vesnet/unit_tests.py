# tests for calc_metrics() in lossfunctions.py

import copy
import os
import numpy as np

import unittest
import math

import nibabel as nib
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



# modules to get tested
from lossfunctions import calc_metrics
from dataloader import DataAugmentation

from dataloader import RSOMVesselDataset
from dataloader import DropBlue, ToTensor, to_numpy
from patch_handling import get_patches, get_volume


os.environ["CUDA_VISIBLE_DEVICES"]='5'

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

    if x.sum() + y.sum() == 0:
        return 1.

    return (2. * i.sum()) / (x.sum() + y.sum())

def isclose(a, b):
    return math.isclose(a, b, rel_tol=1e-07)

def saveNII(V, path):
        img = nib.Nifti1Image(V, np.eye(4))
        nib.save(img, str(path))

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

        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertEqual(dice, dice__)

        # two pixel vs one pixel
        target[0, 0, 0, 0, 1] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertTrue(isclose(dice, dice__))
        
        # no overlap, zero
        target[0, 0, 0, 0, 0] = 0
        target[0, 0, 0, 0, 1] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertEqual(dice, dice__)

    def test_dice_batch_size_2(self):

        # 2x2x2 volume
        # complete overlap
        pred = torch.zeros((2, 1, 2, 2, 2), dtype=torch.float32)
        pred[0, 0, 0, 0, 0] = 1
        pred = pred.cuda()
        target = copy.deepcopy(pred)
        skel = copy.deepcopy(pred)

        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertEqual(dice, dice__)

        # 1 pixel vs 3 pixel
        target[:, 0, 0, 0, 1] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertTrue(isclose(dice, dice__))
        
        # no overlap, zero
        target[:, 0, 0, 0, :] = 0
        target[1, 0, 0, 0, 0] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        dice = _dice(pred, target)
        self.assertEqual(dice, dice__)

    def test_cl_score_batch_size_1(self):
        # complete overlap, cl score = 1
        pred = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        pred[0, 0, 0, 0, 0] = 1
        pred = pred.cuda()
        target = copy.deepcopy(pred)
        skel = copy.deepcopy(pred)

        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(1., cl_score)

        #skeleton enclosed
        pred[0, 0, :, :, :] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(1., cl_score)

        # skeleton outside
        pred[:] = 0
        pred[0, 0, 0, 0, 1] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(0., cl_score)

        # pixel enclosed 2 pixel outside
        # cl_score 0.333
        skel[0, 0, 0, 0, :] = 1
        skel[0, 0, 0, 1, 0] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertTrue(isclose(1/3, cl_score))

    def test_cl_score_batch_size_2(self):
         # complete overlap, cl score = 1
        pred = torch.zeros((2, 1, 2, 2, 2), dtype=torch.float32)
        pred[0, 0, 0, 0, 0] = 1
        pred = pred.cuda()
        target = copy.deepcopy(pred)
        skel = copy.deepcopy(pred)
        
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(1., cl_score)

        #skeleton enclosed
        pred[0, 0, :, :, :] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(1., cl_score)

        # skeleton outside
        pred[:] = 0
        pred[:, 0, 0, 0, 1] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(0., cl_score)

        # 2 pixel enclosed 6 pixel outside
        # cl_score 1/3
        skel[:, 0, 0, 0, :] = 1
        skel[:, 0, 0, 1, 0] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertTrue(isclose(1/3, cl_score))

    def test_out_score_batch_size_1(self):
        
        # hull encludes everything but not corner elements, eg [0,0,0,0,0]
        target = torch.zeros((1, 1, 7, 7, 7), dtype=torch.float32)
        target[0, 0, 3, 3, 3] = 1
        target = target.cuda()
        skel = copy.deepcopy(target)
        
        # same, outside score = 1
        pred = copy.deepcopy(target)
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(1., out_score)
    
        # pred completely outside 
        pred = torch.zeros((1, 1, 7, 7, 7), dtype=torch.float32)
        pred = pred.cuda()
        pred[0, 0, 0, 0, 0] = 1
        pred[0, 0, 0, 0, 6] = 1
        pred[0, 0, 0, 6, 0] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(0., out_score)

        # pred partially inside
        pred = torch.zeros((1, 1, 7, 7, 7), dtype=torch.float32)
        pred = pred.cuda()
        pred[0, 0, 0, 0, 5:7] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(0.5, out_score)

    def test_out_score_batch_size_2(self):
        # hull encludes everything but not corner elements, eg [0,0,0,0,0]
        target = torch.zeros((2, 1, 7, 7, 7), dtype=torch.float32)
        target[0, 0, 3, 3, 3] = 1
        target = target.cuda()
        skel = copy.deepcopy(target)
        
        # same, outside score = 1
        pred = copy.deepcopy(target)
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(1., out_score)
    
        # pred completely outside 
        pred = torch.zeros((2, 1, 7, 7, 7), dtype=torch.float32)
        pred = pred.cuda()
        pred[0, 0, 0, 0, 0] = 1
        pred[0, 0, 0, 0, 6] = 1
        pred[0, 0, 0, 6, 0] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(0., out_score)

        # pred partially inside
        pred = torch.zeros((2, 1, 7, 7, 7), dtype=torch.float32)
        pred = pred.cuda()
        pred[0, 0, 0, 0, 5:7] = 1
        cl_score, out_score, dice__ = calc_metrics(pred, target, skel)
        self.assertEqual(0.5, out_score)

# dataloader
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

    def test_not_rsom(self):
        data = np.random.randint(0, 255, size=(10, 9, 8, 1))
        label = np.random.randint(0, 255, size=(10, 9, 8, 1))
        meta = {'filename': '25.nii.gz'}
        sample = {'data': data, 'label': label, 'meta': meta}

        tf = DataAugmentation()
        out = tf(sample)
        self.assertTrue(np.all(data == out['data']))
        self.assertTrue(np.all(label == out['label']))

class TestDataloaderPatches(unittest.TestCase):

    def setUp(self):
        # 2. generate test directory
        cwd = os.getcwd()
        self.testdir = os.path.join(cwd,'temp_test_dl')
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)
        os.mkdir(self.testdir)
 
    
    def test_patch_reconstruction(self):
        # 1. generate random data and label files
        L_dim = D_dim = (100, 100, 100)
        
        D = [None, None, None]
        L = [None, None, None]
        Dname = [None, None, None]
        Lname = [None, None, None]
        
        D[0] = np.random.random_sample(D_dim)
        D[1] = np.random.random_sample(D_dim)
        D[2] = np.random.random_sample(D_dim)
        
        L[0] = np.random.random_sample(L_dim)
        L[1] = np.random.random_sample(L_dim)
        L[2] = np.random.random_sample(L_dim)
        
        D[0] = D[0].astype(dtype=np.float32)
        D[1] = D[1].astype(dtype=np.float32)
        D[2] = D[2].astype(dtype=np.float32)
        
        L[0] = L[0].astype(dtype=np.float32)
        L[1] = L[1].astype(dtype=np.float32)
        L[2] = L[2].astype(dtype=np.float32)
        
       
        # 3. save files to test directory
        Dname[0] = '1_v_rgb.nii.gz'
        Dname[1] = '2_v_rgb.nii.gz'
        Dname[2] = '3_v_rgb.nii.gz'
        
        Lname[0] = '1_v_l.nii.gz'
        Lname[1] = '2_v_l.nii.gz'
        Lname[2] = '3_v_l.nii.gz'
        
        saveNII(D[0], os.path.join(self.testdir, Dname[0]))
        saveNII(D[1], os.path.join(self.testdir, Dname[1]))
        saveNII(D[2], os.path.join(self.testdir, Dname[2]))
        
        saveNII(L[0], os.path.join(self.testdir, Lname[0]))
        saveNII(L[1], os.path.join(self.testdir, Lname[1]))
        saveNII(L[2], os.path.join(self.testdir, Lname[2]))
        
        # 4. construct dataset and dataloader
        
        divs = (2, 3, 5)
        offset = (0, 0, 0)
         
        # TODO transforms
        set1 = RSOMVesselDataset(self.testdir, 
                                 divs=divs, 
                                 offset = offset,
                                 transform=transforms.Compose([ToTensor()]))
        
        dataloader = DataLoader(set1,
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=1, 
                                pin_memory=True)
        
        
        
        # 5. draw samples and reconstruct the patches to volumes.
        #try:
        if 1:
            set1_iter = iter(dataloader)
            rem = []
            
            # 3 files
            for file in np.arange(len(Dname)):
                Dout = []
                Lout = []
                
                # prod(divs) patches in each sample
                for ctr in np.arange(np.prod(divs)):
                    patch = next(set1_iter)
                    #print('INDEX:', patch['meta']['index'])
                    #print('filename:',patch['meta']['filename'])
                    #print(patch['data'].shape)
                    Dout.append(patch['data'].squeeze())
                    Lout.append(patch['label'].squeeze())
                
                if isinstance(patch['data'], torch.Tensor):
                    Dout = (torch.stack(Dout)).numpy()
                    Lout = (torch.stack(Lout)).numpy()
                else:
                    Dout = np.array(Dout)
                    Lout = np.array(Lout)
                
                # print('Dout, Lout shapes:', Dout.shape, Lout.shape)
                Dvol_ = get_volume(Dout, divs, offset)
                Lvol_ = get_volume(Lout, divs, offset)
                    
                # print('Dvol, Lvol shapes:', Dvol_.shape, Lvol_.shape)
                
                Dvol = to_numpy(Dvol_, 
                                patch['meta'],
                                Vtype='data',
                                dimorder='torch')
                Lvol = to_numpy(Lvol_, 
                                patch['meta'],
                                Vtype='label',
                                dimorder='torch')
                
                rem.append(Dvol==D[Dname.index(patch['meta']['filename'][0])])
                
                # 6. compare to generated data.
                
                Dbool = Dvol == D[Dname.index(patch['meta']['filename'][0])]
                Lbool = Lvol == L[Dname.index(patch['meta']['filename'][0])]
                
                # apply crop what was reconstructed with zeros
                b = patch['meta']['dcrop']['begin'].numpy()[0]
                e = patch['meta']['dcrop']['end'].numpy()[0]
                if e[0] == 0:
                    Dbool = Dbool[b[0]:,...]
                    Lbool = Lbool[b[0]:,...]
                else:
                    Dbool = Dbool[b[0]:e[0],...]
                    Lbool = Lbool[b[0]:e[0],...]
                if e[1] == 0:
                    Dbool = Dbool[:, b[0]:,...]
                    Lbool = Lbool[:, b[0]:,...]
        
                else:
                    Dbool = Dbool[:, b[0]:e[0],...]  
                    Lbool = Lbool[:, b[0]:e[0],...]              
                if e[2] == 0:
                    Dbool = Dbool[...,b[0]:]
                    Lbool = Lbool[...,b[0]:]
                else:
                    Dbool = Dbool[...,b[0]:e[0]]  
                    Lbool = Lbool[...,b[0]:e[0]]          
                self.assertTrue(np.all(Dbool))
                self.assertTrue(np.all(Lbool))

    def tearDown(self):
        shutil.rmtree(self.testdir)

# patch_handling
class TestPatchHandling(unittest.TestCase):
    def _testit(self, A, divs, offset):
        A_p = get_patches(A, divs, offset)
        A_ = get_volume(A_p, divs, offset)

        if A_.shape == A.shape:
            if np.all(A_ == A):
                return True
            else:
                return False
        else:
            return False


    def test_trivial(self):
        V = np.random.random_sample((100, 100, 100))
        self.assertTrue(self._testit(V,(1,1,1),(0,0,0)))

        V = np.random.random_sample((1, 1, 1))
        self.assertTrue(self._testit(V,(1,1,1),(0,0,0)))

        V = np.random.random_sample((1, 3))
        self.assertTrue(self._testit(V,(1),(0)))

        V = np.random.random_sample((1))
        self.assertTrue(self._testit(V,(1),(0)))

        V = np.random.random_sample((1))
        self.assertTrue(self._testit(V,(1),(5)))

    def test_1d(self):
        # TEST 1D
        V = np.random.random_sample((100))
        self.assertTrue(self._testit(V, 2, 6))
        self.assertTrue(self._testit(V, 100, 0))
        self.assertTrue(self._testit(V, 50, 10))
        self.assertTrue(self._testit(V, 2, 9))

    def test_2d(self):

        # TEST 2D
        V = np.random.random_sample((100, 100))
        self.assertTrue(self._testit(V,(2,2),(3,3)))
        self.assertTrue(self._testit(V,(2,4),(9,0)))

        V = np.random.random_sample((40, 60))
        self.assertTrue(self._testit(V,(2,2),(3,3)))
        self.assertTrue(self._testit(V,(2,4),(9,0)))

        # TEST 2D RGB
        V = np.random.random_sample((100, 100, 3))
        self.assertTrue(self._testit(V,(2,2),(3,3)))
        self.assertTrue(self._testit(V,(2,4),(9,0)))

    def test_3d(self):
        # TEST 3D
        V = np.random.random_sample((100, 100, 100))
        self.assertTrue(self._testit(V,(2,2,2),(3,3,3)))
        self.assertTrue(self._testit(V,(2,4,10),(9,0,7)))

        V = np.random.random_sample((40, 60, 30))
        self.assertTrue(self._testit(V,(2,2,2),(3,3,3)))
        self.assertTrue(self._testit(V,(2,4,5),(9,0,1)))
        self.assertTrue(self._testit(V,(2,2,1),(3,3,3)))

        # TEST 3D with fake singleton dimension
        V = np.random.random_sample((100, 100, 100, 1))
        self.assertTrue(self._testit(V,(2,2,2),(3,3,3)))
        self.assertTrue(self._testit(V,(2,4,5),(9,0,7)))

        V = np.random.random_sample((40, 60, 30, 1))
        self.assertTrue(self._testit(V,(2,2,2),(3,3,3)))
        self.assertTrue(self._testit(V,(2,4,5),(9,0,1)))
        self.assertTrue(self._testit(V,(2,2,1),(3,3,3)))

        # TEST 3D RGB
        V = np.random.random_sample((100, 100, 100, 3))
        self.assertTrue(self._testit(V,(2,2,2),(3,3,3)))
        self.assertTrue(self._testit(V,(2,4,2),(9,0,7)))

        V = np.random.random_sample((40, 60, 30, 3))
        self.assertTrue(self._testit(V,(2,2,2),(3,3,3)))
        self.assertTrue(self._testit(V,(2,4,5),(9,0,1)))
        self.assertTrue(self._testit(V,(2,2,1),(3,3,3)))

    def test_4d(self):
        # TEST 4D
        V = np.random.random_sample((100, 100, 100, 100))
        self.assertTrue(self._testit(V,(2,2,2,2),(3,3,3,3)))
        self.assertTrue(self._testit(V,(2,4,5,2),(9,3,7,3)))

        V = np.random.random_sample((40, 60, 30, 100))
        self.assertTrue(self._testit(V,(2,2,2,2),(3,3,3,3)))
        self.assertTrue(self._testit(V,(2,4,5,2),(9,0,1,5)))

    def test_random(self):
        # totally random
        for _ in np.arange(10):
            Ndim = np.random.randint(2,7)
            MultiChannel = np.random.randint(2)
            divs = np.random.randint(1,4,size=Ndim-MultiChannel)
            multipliers =  np.random.randint(1,4,size=Ndim-MultiChannel)
            offset = tuple(np.random.randint(1,10,size=Ndim-MultiChannel))
            dimensions = divs*multipliers
            # print('Ndim',Ndim, 'MultiChannel?', bool(MultiChannel))
            # print('divs:', divs, 'offset:', offset, 'dimensions:', dimensions)
            V = np.random.random_sample(tuple(dimensions))
            self.assertTrue(self._testit(V, divs, offset))





suiteList = []
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestCalcMetrics))
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestDataloaderDataAugmentation))
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestDataloaderPatches))
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestPatchHandling))

suite = unittest.TestSuite(suiteList)
unittest.TextTestRunner(verbosity=2).run(suite)



















