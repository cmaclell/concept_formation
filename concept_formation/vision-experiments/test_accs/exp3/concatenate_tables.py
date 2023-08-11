import os
import pandas as pd

# Folder path containing CSV files
folder_path = os.getcwd()  # Replace with the path to your folder

# Get a list of CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to hold DataFrames
dataframes = []

# Read each CSV file into a DataFrame and add it to the list
for csv_file in csv_files:
    csv_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(csv_path)
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
output_file = 'exp3.csv'
concatenated_df.to_csv(output_file, index=False)

print('CSV files concatenated and saved as', output_file)


