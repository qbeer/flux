import unittest
from flux.datasets.vision.vqa import VQA

class VQATests(unittest.TestCase):
    """Tests for `vqa.py`."""

    def __init__(self, *args, **kwargs):
        self.vqa = VQA()

    def test_crash_test(self):
        val_db = self.vqa.val_db
        # make sure valdb is built
        self.assertTrue(val_db is not None)

    def test_sample_test(self):
        image, question, answer = self.vqa.data_format
        self.assertTrue(image is not None)
        self.assertTrue(question is not None)
        self.assertTrue(answer is not None)

    # def sample_test(self):
    #     self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()