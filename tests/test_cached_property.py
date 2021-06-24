#!/USSR/bin/python3.9
""" Script to show lab environment on computer """
__author__ = "LukasHuber"
__date__ = "2021-05-18"
__email__ = "lukas.huber@epfl.ch"

import unittest

import random
import copy

from functools_custom import lru_cached_property

class TestSum(unittest.TestCase):
    def test_creation(self):
        class TestCreation():
            def __init__(self):
                self.value1 = 1
                self.value2 = 2
                self.value3 = 3

                self.iterator = 0
                self.dummy_value = 0

            @lru_cached_property(arg_list=['value1', 'value2', 'value3'], maxsize=1)
            def prop_incremental(self):
                self.iterator += 1
                # print('New iterator', self.iterator)
                return self.value1 + self.value2

        Instance = TestCreation()
        val = Instance.prop_incremental
        it_start = copy.deepcopy(Instance.iterator)
        for ii in range(10):
            val = Instance.prop_incremental
            self.assertEqual(it_start, Instance.iterator, 'Lru cache was called again')

        # Check when adapting values 1
        it_start = copy.deepcopy(Instance.iterator)
        Instance.value1 = random.randint(0, 1000)
        for ii in range(10):
            val = Instance.prop_incremental
            self.assertEqual(it_start, Instance.iterator - 1, 'LRU-cache was not only called once.')

        # Check when adapting values 2
        for ii in range(10):
            it_start = copy.deepcopy(Instance.iterator)
            new_value = random.randint(0, 1000)
            if new_value == Instance.value2:
                continue
            Instance.value2 = new_value
            val = Instance.prop_incremental
            self.assertEqual(it_start, Instance.iterator - 1, 'LRU-cache was not only called once.')

        # Check when modifying dummy values
        it_start = copy.deepcopy(Instance.iterator)
        for ii in range(10):
            Instance.dummy_value = random.randint(0, 100)
            val = Instance.prop_incremental
            self.assertEqual(it_start, Instance.iterator, 'LRU-cache was wrongly called.')


    def test_check_cache(self):
        vals_cache = [0, 1, 2]
        vals_nocache = [3, 4, 5]
        vals_new = [6, 7, 8, 9]

        cache_size = len(vals_cache)
        class TestCreation():
            def __init__(self):
                self.value1 = 100
                self.value2 = 200
                self.value3 = 300

                self.iterator = 0
                self.dummy_value = 0

            @lru_cached_property(arg_list=['value1', 'value2', 'value3'], maxsize=cache_size)
            def prop_incremental(self):
                self.iterator += 1
                # print('New iterator', self.iterator)
                return self.value1 + self.value2

        Instance = TestCreation()

        # These values will not stay in the cache
        for ii in range(len(vals_nocache)):
            Instance.value1 = vals_nocache[ii]
            result = Instance.prop_incremental

        # Values which will stay in the cache
        for ii in range(cache_size):
            Instance.value1 = vals_cache[ii]
            result = Instance.prop_incremental

        # Function should not be evaluated with values left in the cache
        it_start = copy.deepcopy(Instance.iterator)
        for ii in range(cache_size):
            Instance.value1 = vals_cache[ii]
            result = Instance.prop_incremental
            self.assertEqual(it_start, Instance.iterator, 'LRU-cache was wrongly called.')

        # Function should be evaluated with new values (removed from cache)
        for ii in range(len(vals_nocache)):
            it_start = copy.deepcopy(Instance.iterator)
            Instance.value1 = vals_nocache[ii]
            result = Instance.prop_incremental
            self.assertEqual(it_start, Instance.iterator-1, 'LRU-cache was not called.')

        # Function should be evaluated with new values (removed from cache)
        for ii in range(len(vals_new)):
            it_start = copy.deepcopy(Instance.iterator)
            Instance.value1 = vals_new[ii]
            result = Instance.prop_incremental
            self.assertEqual(it_start, Instance.iterator-1, 'LRU-cache was not called.')

if __name__ == '__main__':
    unittest.main()
    
print('Done')
