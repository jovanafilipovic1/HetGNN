import glob
import csv # Keep for potential fallback or if user wants to revert, though pandas is primary
import os
import pandas as pd # Added pandas

def analyze_training_times(results_dir="Results"):
    """
    Analyzes training times (converted to minutes) from grid_results.csv files 
    within specified subdirectories using pandas.
    Prints min, max, and mean training time for each file and globally.

    Args:
        results_dir (str): The base directory containing the results.
                           Defaults to "Results".

    Returns:
        None: Prints the statistics or messages if no data is found.
    """
    search_pattern = os.path.join(results_dir, "Neurobl*", "grid_search", "grid_results.csv")
    all_training_times_minutes = [] # Changed variable name for clarity

    print(f"Searching for files matching: {search_pattern}")
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        print("No matching 'grid_results.csv' files found.")
        return

    print(f"\nFound {len(file_paths)} files. Processing each file:")
    print("-" * 40) # Separator

    for file_path in file_paths:
        print(f"\nProcessing File: {file_path}")
        try:
            df = pd.read_csv(file_path)
            
            if "training_time" not in df.columns:
                print(f"  Warning: 'training_time' column not found. Skipping this file.")
                print("-" * 40) # Separator
                continue
            
            # Convert training time to numeric, then to minutes, drop NaNs
            training_times_for_file_minutes = (pd.to_numeric(df["training_time"], errors='coerce') / 60).dropna().tolist()
            
            if not training_times_for_file_minutes:
                print(f"  Warning: No valid numeric 'training_time' data found in this file after conversion. Skipping.")
                print("-" * 40) # Separator
                continue

            # Calculate and print stats for the current file
            min_time_file = min(training_times_for_file_minutes)
            max_time_file = max(training_times_for_file_minutes)
            mean_time_file = sum(training_times_for_file_minutes) / len(training_times_for_file_minutes)
            
            print(f"  Statistics for this file ({len(training_times_for_file_minutes)} data points):")
            print(f"    Min Training Time:  {min_time_file:.2f} minutes")
            print(f"    Max Training Time:  {max_time_file:.2f} minutes")
            print(f"    Mean Training Time: {mean_time_file:.2f} minutes")
            
            all_training_times_minutes.extend(training_times_for_file_minutes)
            
        except FileNotFoundError:
            print(f"  Error: File not found. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"  Warning: File is empty. Skipping.")
        except Exception as e:
            print(f"  An unexpected error occurred while processing with pandas: {e}. Skipping.")
        finally:
            print("-" * 40) # Separator after each file processing block


    if not all_training_times_minutes:
        print("\nNo overall training_time data found in any of the processed files or data was not numeric.")
        return

    # Global statistics
    global_min_time = min(all_training_times_minutes)
    global_max_time = max(all_training_times_minutes)
    global_mean_time = sum(all_training_times_minutes) / len(all_training_times_minutes)

    print("\n--- Overall Training Time Statistics (across all files) ---")
    print(f"Minimum Training Time: {global_min_time:.2f} minutes")
    print(f"Maximum Training Time: {global_max_time:.2f} minutes")
    print(f"Mean Training Time:    {global_mean_time:.2f} minutes")
    print(f"Total valid data points from all files: {len(all_training_times_minutes)}")

if __name__ == "__main__":
    # Assuming the script is run from the root of the GIT_HetGNN directory
    # or that 'Results' is in the current working directory.
    analyze_training_times() 