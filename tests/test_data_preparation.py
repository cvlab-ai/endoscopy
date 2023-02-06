import main as program
import unittest

from tests.file_comperer import are_dir_trees_equal


class TestDataPreparation(unittest.TestCase):

    def test_binary_seg(self):
        program.main(
            [
                "--ers-path",
                "ers",
                "--ers-class-mapper-path",
                "binary-seg/2-class.yaml",
                "--ers-use-seq",
                "--ers-use-empty-masks",
                "--training-type",
                "binary-seg",
                "--train-size",
                "1",
                "-f",
                "--output-path",
                "binary-seg/data"
            ]
        )
        result = are_dir_trees_equal("binary-seg/data", "binary-seg/expected_data")
        self.assertTrue(result)

    def test_multilabel_seg(self):
        program.main(
            [
                "--ers-path",
                "ers",
                "--ers-class-mapper-path",
                "multilabel-seg/2-class.yaml",
                "--ers-use-seq",
                "--ers-use-empty-masks",
                "--training-type",
                "multilabel-seg",
                "--train-size",
                "1",
                "-f",
                "--output-path",
                "multilabel-seg/data"
            ]
        )
        result = are_dir_trees_equal("multilabel-seg/data", "multilabel-seg/expected_data")
        self.assertTrue(result)

    def test_multilabel_classification(self):
        program.main(
            [
                "--ers-path",
                "ers",
                "--ers-class-mapper-path",
                "multilabel-classification/4-class.yaml",
                "--ers-use-empty-masks",
                "--training-type",
                "multilabel-classification",
                "--train-size",
                "1",
                "-f",
                "--output-path",
                "multilabel-classification/data"
            ]
        )
        result = are_dir_trees_equal("multilabel-classification/data", "multilabel-classification/expected_data")
        self.assertTrue(result)
