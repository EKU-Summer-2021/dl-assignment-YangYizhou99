import unittest
from src import grade_run
import os


class DecisionTreeTest(unittest.TestCase):

    def test_function(self):
        grade_run.grade_run()
        self.assertTrue(os.path.exists('grade_result'))