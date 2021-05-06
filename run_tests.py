# run all software tests
import os
import unittest

from vesnet.unittests import TestCalcMetrics, TestDataloaderDataAugmentation, \
                             TestDataloaderPatches, TestPatchHandling



os.environ["CUDA_VISIBLE_DEVICES"]='0'

suiteList = []
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestCalcMetrics))
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestDataloaderDataAugmentation))
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestDataloaderPatches))
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestPatchHandling))

suite = unittest.TestSuite(suiteList)
unittest.TextTestRunner(verbosity=2).run(suite)



















