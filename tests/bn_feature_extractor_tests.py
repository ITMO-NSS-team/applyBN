import pytest
import os


def run_tests():
    test_files = ['fit_tests.py', 'transform_tests.py', 'initialization_test.py']

    for test_file in test_files:
        print(f"test name: {test_file}")
        pytest.main([test_file])


if __name__ == "__main__":
    run_tests()