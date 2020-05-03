import os
import unittest

from clane.graphs.utils import get_graph_dataset


class TestDataLoading(unittest.TestCase):
    def test_relative_path_input(self):
        """
            Test that it can load data, when the input path is RELATIVE path.
        """
        path = './data/cora_custom'
        graph = get_graph_dataset(path)
        self.assertEqual(graph.num_nodes, 2708)

    def test_absolute_path_input(self):
        """
            Test that it can load data, when the input path is ABSOLUTE path.
        """
        path = './data/cora_custom'
        path = os.path.abspath(path)
        graph = get_graph_dataset(path)
        self.assertEqual(graph.num_nodes, 2708)

    def test_wrong_path_input(self):
        """
            Test that it can raise an exception, when the input path not exsits.
        """
        path = './data/NOTEXISTING_PATH'
        with self.assertRaises(FileNotFoundError):
            graph = get_graph_dataset(path)


if __name__ == "__main__":
    unittest.main()
