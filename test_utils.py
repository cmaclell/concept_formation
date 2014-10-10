import utils
import unittest
import random

class TestUtils(unittest.TestCase):

    def test_cv_mean(self):
        for i in range(10):
            values = [random.normalvariate(0, 1) for i in range(100)] 
            cv = utils.ContinuousValue()
            cv.update_batch(values) 
            assert cv.mean - utils.mean(values) < 0.00000000001

    def test_cv_std(self):
        for i in range(10):
            values = [random.normalvariate(0, 1) for i in range(100)] 
            cv = utils.ContinuousValue()
            cv.update_batch(values) 
            assert cv.biased_std() - utils.std(values) < 0.00000000001

    def test_cv_unbiased_std(self):
        for i in range(10):
            values = [random.normalvariate(0, 1) for i in range(100)] 
            cv = utils.ContinuousValue()
            cv.update_batch(values) 
            assert (cv.unbiased_std() - utils.c4(len(values)) * 
                    (len(values) * utils.std(values) / (len(values) - 1)) <
                    0.00000000001)

    def test_cv_update(self):
        for i in range(10):
            values = []
            cv = utils.ContinuousValue()
            for i in range(20):
                x = random.normalvariate(0,1)
                values.append(x)
                cv.update(x)
                assert cv.biased_std() - utils.std(values) < 0.00000000001

    def test_cv_combine(self):
        for i in range(10):
            values1 = [random.normalvariate(0,1) for i in range(50)]
            values2 = [random.normalvariate(0,1) for i in range(50)]
            values = values1 + values2
            cv = utils.ContinuousValue()
            cv2 = utils.ContinuousValue()

            cv.update_batch(values1)
            assert cv.biased_std() - utils.std(values1) < 0.00000000001

            cv2.update_batch(values2)
            assert cv2.biased_std() - utils.std(values2) < 0.00000000001

            cv.combine(cv2)
            assert cv.biased_std() - utils.std(values) < 0.00000000001

if __name__ == "__main__":
    unittest.main()


