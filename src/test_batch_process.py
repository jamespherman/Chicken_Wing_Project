import unittest
from unittest.mock import patch
from batch_process_with_heatmaps import main as batch_main

class TestBatchProcess(unittest.TestCase):
    @patch('builtins.print')
    def test_main_runs_without_errors(self, mock_print):
        # This test checks if the main function runs without raising exceptions.
        # It does not check for correctness of the output.
        try:
            result = batch_main()
            self.assertTrue(result, "main function should return True on success")
        except Exception as e:
            self.fail(f"main function raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
