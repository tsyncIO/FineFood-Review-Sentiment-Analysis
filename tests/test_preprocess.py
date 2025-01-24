import unittest
from src.preprocess import clean_text

class TestPreprocess(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("Hello World!"), "hello world")
        self.assertEqual(clean_text("Visit https://example.com"), "visit")
