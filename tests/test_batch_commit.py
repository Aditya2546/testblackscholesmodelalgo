#!/usr/bin/env python3
"""
Tests for the batch_commit.py script.

Tests the functionality of batch commits by mocking subprocess.run
to simulate file operations without actually making Git changes.
"""

import unittest
from unittest.mock import patch, Mock
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path so we can import the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.batch_commit import batch_commit_files, get_uncommitted_files, chunk_list


class TestBatchCommit(unittest.TestCase):
    """Test cases for batch_commit.py."""

    def setUp(self):
        """Set up test environment."""
        # Create list of mock files
        self.mock_files = [f"file_{i}.py" for i in range(1, 28)]
        
        # Create mock arguments
        self.mock_args = Mock()
        self.mock_args.batch_size = 20
        self.mock_args.remote = "origin"
        self.mock_args.branch = "main"
        self.mock_args.dry_run = False

    @patch('scripts.batch_commit.run_command')
    def test_get_uncommitted_files(self, mock_run_command):
        """Test get_uncommitted_files function with mock data."""
        # Mock git status output
        status_output = "\n".join([f"?? {file}" for file in self.mock_files])
        mock_run_command.return_value = (0, status_output, "")
        
        # Call the function
        result = get_uncommitted_files()
        
        # Verify the function called git status
        mock_run_command.assert_called_once_with(["git", "status", "--porcelain"])
        
        # Verify result contains the expected files
        self.assertEqual(len(result), 27)
        self.assertEqual(result, self.mock_files)

    def test_chunk_list(self):
        """Test the chunk_list function."""
        # Test with batch size 20
        chunks = chunk_list(self.mock_files, 20)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 20)
        self.assertEqual(len(chunks[1]), 7)
        
        # Test with batch size 10
        chunks = chunk_list(self.mock_files, 10)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 10)
        self.assertEqual(len(chunks[1]), 10)
        self.assertEqual(len(chunks[2]), 7)

    @patch('scripts.batch_commit.get_uncommitted_files')
    @patch('scripts.batch_commit.run_command')
    def test_batch_commit_files(self, mock_run_command, mock_get_uncommitted_files):
        """Test batch_commit_files with mock data (27 files, batch size 20)."""
        # Mock uncommitted files
        mock_get_uncommitted_files.return_value = self.mock_files
        
        # Mock successful command executions
        mock_run_command.return_value = (0, "success", "")
        
        # Call the function
        result = batch_commit_files(self.mock_args)
        
        # Verify the result is successful
        self.assertEqual(result, 0)
        
        # Check that run_command was called the expected number of times
        # Each batch requires 3 commands: git add, git commit, git push
        # 2 batches = 6 calls total
        self.assertEqual(mock_run_command.call_count, 6)
        
        # Check commits were done with correct messages
        commit_calls = [
            call for call in mock_run_command.call_args_list 
            if call[0][0][0:2] == ["git", "commit"]
        ]
        self.assertEqual(len(commit_calls), 2)
        self.assertIn("1/2", commit_calls[0][0][0][3])
        self.assertIn("2/2", commit_calls[1][0][0][3])

    @patch('scripts.batch_commit.get_uncommitted_files')
    @patch('scripts.batch_commit.run_command')
    def test_failed_push(self, mock_run_command, mock_get_uncommitted_files):
        """Test batch_commit_files with a failed push."""
        # Mock uncommitted files
        mock_get_uncommitted_files.return_value = self.mock_files
        
        # Mock command executions, making the first push fail
        def side_effect(command, dry_run=False):
            if command[0:2] == ["git", "push"] and command[2:4] == ["origin", "main"]:
                return (1, "", "remote: error: too many files")
            return (0, "success", "")
        
        mock_run_command.side_effect = side_effect
        
        # Call the function
        result = batch_commit_files(self.mock_args)
        
        # Verify the result shows failure
        self.assertEqual(result, 1)
        
        # Check that we attempted add, commit, and push but stopped after first failed push
        self.assertEqual(mock_run_command.call_count, 3)


if __name__ == "__main__":
    unittest.main() 