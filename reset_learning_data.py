#!/usr/bin/env python
"""
Reset Learning Data

This script resets the machine learning data for the Smart Trader system.
Run this when switching between simulated and real market data to ensure
the system learns patterns specific to the current data source.
"""

import os
import json
import datetime
import shutil
from smart_trader import SmartTrader
from realistic_smart_trader import RealisticSmartTrader

def reset_learning_data(data_source="Simulated"):
    """
    Reset the machine learning data for the Smart Trader system.
    
    Args:
        data_source: The data source to reset for ("Simulated" or "Alpaca Real-Time")
    """
    source_tag = "real" if data_source == "Alpaca Real-Time" else "sim"
    
    # Paths to pattern files for current data source
    pattern_files = [
        f"smart_trader_patterns_{source_tag}.json",
        f"realistic_smart_trader_patterns_{source_tag}.json"
    ]
    
    # Also handle the original pattern files (for backward compatibility)
    legacy_files = [
        "smart_trader_patterns.json",
        "realistic_smart_trader_patterns.json"
    ]
    
    # Files from the other data source that should be backed up but not active
    other_source_tag = "sim" if data_source == "Alpaca Real-Time" else "real"
    other_source_files = [
        f"smart_trader_patterns_{other_source_tag}.json",
        f"realistic_smart_trader_patterns_{other_source_tag}.json"
    ]
    
    # Create backup directory if it doesn't exist
    backup_dir = "ml_backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Current timestamp for backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # First, backup and remove files from other data source
    for pattern_file in other_source_files:
        if os.path.exists(pattern_file):
            print(f"Found pattern file from other data source: {pattern_file}")
            
            # Create backup
            backup_file = f"{backup_dir}/{timestamp}_{pattern_file}"
            shutil.copy2(pattern_file, backup_file)
            print(f"Created backup: {backup_file}")
            
            # Remove the file completely
            os.remove(pattern_file)
            print(f"Removed pattern file: {pattern_file}")
    
    # Now process pattern files for the current data source
    all_files = pattern_files + legacy_files
    
    for pattern_file in all_files:
        if os.path.exists(pattern_file):
            print(f"Found existing pattern file: {pattern_file}")
            
            # Create backup
            backup_file = f"{backup_dir}/{timestamp}_{pattern_file}"
            shutil.copy2(pattern_file, backup_file)
            print(f"Created backup: {backup_file}")
            
            # Reset pattern file to empty
            with open(pattern_file, 'w') as f:
                json.dump({
                    "patterns": {}, 
                    "metadata": {
                        "created": timestamp, 
                        "source": data_source,
                        "reset_reason": "manual_reset"
                    }
                }, f)
            print(f"Reset pattern file: {pattern_file}")
        else:
            print(f"Pattern file not found: {pattern_file}")
            # Create an empty file
            with open(pattern_file, 'w') as f:
                json.dump({
                    "patterns": {}, 
                    "metadata": {
                        "created": timestamp, 
                        "source": data_source,
                        "reset_reason": "new_file_creation"
                    }
                }, f)
            print(f"Created new empty pattern file: {pattern_file}")
    
    # Initialize traders to create fresh pattern files
    print("Initializing traders with fresh pattern files...")
    
    # Set environment variable to control which pattern files to use
    os.environ["TRADING_DATA_SOURCE"] = data_source
    
    # Initialize standard trader
    trader = SmartTrader(account_value=25000.0)
    trader.patterns.save()  # Force create pattern file
    
    # Initialize realistic trader
    realistic_trader = RealisticSmartTrader(account_value=25000.0)
    realistic_trader.patterns.save()  # Force create pattern file
    
    print(f"\nLearning data reset complete for {data_source}! The system will now learn from scratch.")
    print(f"All pattern files from other data sources have been backed up and removed.")
    print("Only current data source patterns will be used.")

if __name__ == "__main__":
    print("This will reset all machine learning data for the Smart Trader system.")
    print("Any learned patterns will be backed up but removed from the active system.")
    print("The system will need to learn from scratch with new data.\n")
    
    # Ask for which data source to reset
    print("Select the data source to reset:")
    print("1. Simulated")
    print("2. Alpaca Real-Time")
    print("3. Both")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        data_source = "Simulated"
    elif choice == "2":
        data_source = "Alpaca Real-Time"
    elif choice == "3":
        data_source = "both"
    else:
        print("Invalid choice. Operation cancelled.")
        exit()
    
    confirm = input(f"Are you sure you want to reset learning data for {data_source}? (y/n): ")
    
    if confirm.lower() == 'y':
        if data_source == "both":
            reset_learning_data("Simulated")
            reset_learning_data("Alpaca Real-Time")
        else:
            reset_learning_data(data_source)
    else:
        print("Operation cancelled. No changes were made.") 