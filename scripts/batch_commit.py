#!/usr/bin/env python3
"""
Batch Commit Script

This script helps safely commit multiple files in batches to avoid hitting
GitHub push caps or editor limits. It chunks uncommitted files into batches,
commits them with sequential messages, and pushes each batch to the remote.

Usage:
    python scripts/batch_commit.py [--batch-size N] [--remote NAME] [--branch NAME] [--dry-run]

Arguments:
    --batch-size N   Number of files per batch (default: 15)
    --remote NAME    Git remote to push to (default: origin)
    --branch NAME    Git branch to push to (default: current branch)
    --dry-run        Show what would be done without making changes

Exit codes:
    0 - Success
    1 - Partial failure (see batch_push_fail.log)
"""

import argparse
import logging
import os
import shlex
import subprocess
import sys
from typing import List, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_push_fail.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch commit files safely")
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=15,
        help="Number of files per batch (default: 15)"
    )
    parser.add_argument(
        "--remote", 
        type=str, 
        default="origin",
        help="Git remote to push to (default: origin)"
    )
    parser.add_argument(
        "--branch", 
        type=str, 
        default="",
        help="Git branch to push to (default: current branch)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    return parser.parse_args()


def run_command(command: List[str], dry_run: bool = False) -> Tuple[int, str, str]:
    """
    Run a command securely and return its exit code, stdout, and stderr.
    
    Args:
        command: List of command components
        dry_run: If True, only print the command without executing it
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    cmd_str = ' '.join(shlex.quote(c) for c in command)
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {cmd_str}")
        return 0, "", ""
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        logger.error(f"Command failed: {cmd_str}")
        logger.error(f"Error: {str(e)}")
        return 1, "", str(e)


def get_current_branch() -> str:
    """Get the name of the current git branch."""
    code, stdout, stderr = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if code != 0:
        logger.error(f"Failed to get current branch: {stderr}")
        sys.exit(1)
    return stdout.strip()


def get_uncommitted_files() -> List[str]:
    """Get a list of all uncommitted files (untracked + modified)."""
    code, stdout, stderr = run_command(["git", "status", "--porcelain"])
    if code != 0:
        logger.error(f"Failed to get git status: {stderr}")
        sys.exit(1)
    
    files = []
    for line in stdout.splitlines():
        if line:
            # Extract the filename from the status output
            # Format is typically "XY path" where X and Y are status codes
            status = line[:2]
            filename = line[3:].strip()
            
            # Handle renamed files (R)
            if status.startswith('R'):
                # For renamed files, the format is "R path/old -> path/new"
                filename = filename.split(' -> ')[1]
            
            # Skip submodules (typically shown with '160000')
            if '160000' not in line:
                files.append(filename)
    
    return files


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def batch_commit_files(args: argparse.Namespace) -> int:
    """
    Main function to batch commit and push files.
    
    Returns:
        0 for success, 1 for partial failure
    """
    # Get branch name (use current if not specified)
    branch = args.branch if args.branch else get_current_branch()
    
    # Get list of uncommitted files
    uncommitted_files = get_uncommitted_files()
    
    if not uncommitted_files:
        print("No uncommitted files found.")
        return 0
    
    # Split files into batches
    batches = chunk_list(uncommitted_files, args.batch_size)
    total_batches = len(batches)
    
    print(f"Found {len(uncommitted_files)} uncommitted files - will commit in {total_batches} batches")
    
    if args.dry_run:
        for i, batch in enumerate(batches, 1):
            print(f"\nBatch {i}/{total_batches} ({len(batch)} files):")
            for file in batch:
                print(f"  {file}")
        print("\nDry run complete. No changes were made.")
        return 0
    
    # Process each batch
    for i, batch in enumerate(batches, 1):
        print(f"\nProcessing batch {i}/{total_batches} ({len(batch)} files)")
        
        # Add files
        code, _, stderr = run_command(["git", "add"] + batch)
        if code != 0:
            logger.error(f"Failed to add files in batch {i}: {stderr}")
            return 1
        
        # Commit
        commit_msg = f"Batch commit {i}/{total_batches}"
        code, _, stderr = run_command(["git", "commit", "-m", commit_msg])
        if code != 0:
            logger.error(f"Failed to commit batch {i}: {stderr}")
            return 1
        
        # Push
        code, _, stderr = run_command(["git", "push", args.remote, branch])
        if code != 0:
            logger.error(f"Failed to push batch {i} to {args.remote}/{branch}: {stderr}")
            logger.error("Aborting further operations. Fix push issues and run again.")
            return 1
        
        print(f"Completed batch {i}/{total_batches}")
    
    print("\n✅ All files committed.")
    return 0


def main():
    """Main entry point."""
    args = parse_arguments()
    exit_code = batch_commit_files(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 