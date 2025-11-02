import pandas as pd
import os

# Explanation: This script processes a CSV file with columns: Filename, paragraph, Topic, Original_Doc_Index.
# It filters rows based on a given set of topics, then for each matching row, it generates a unique text file
# in a common folder named 'extracted_paragraphs'. The text file name is constructed from the original 'Filename'
# (without .txt extension), appended with a generated name for the paragraph. The generated name is 'para_{Original_Doc_Index}_{row_index}',
# where row_index is the pandas DataFrame index for uniqueness (in case multiple paragraphs share the same Original_Doc_Index).
# This ensures no file overwrites if multiple paragraphs come from the same document.
# The paragraph content from the row is written to this new text file.
# Assumptions:
# - The CSV file is named 'input.csv' and is in the same directory as this script.
# - Topics are provided as a list; you can modify this list as needed.
# - The folder 'extracted_paragraphs' will be created if it doesn't exist.
# - Handles quoting in CSV properly using pandas.

# Step 1: Define the input CSV file path and the output folder.
# Get the base path
USAWC = os.getenv('USAWC')
csv_file = f"{USAWC}\\Desktop\\Data\\USAWC_EDU\\Research\\SQRL\\SRR\\Results\\BERTopic\\BERTopic_Results_Paragraph_Level_20251019_143631_no_annotations\\paragraph_topic_assignments.csv"
output_folder = f"{USAWC}\\Desktop\\Data\\USAWC_EDU\\Research\\SQRL\\SRR\\Results\\BERTopic\\extracted_warfighting_paragraphs_no_outliers"

# Step 2: Define the set of topics to filter on. This can be modified or passed as input.
# For example, topics = [21, 22] to include multiple topics.
topics = [6, 21, 28, 35, 38, 39]  # Example: Filtering for topic 21 based on the sample data.

# Step 3: Read the CSV file into a pandas DataFrame.
# pandas handles quoted fields and commas inside quotes automatically.
df = pd.read_csv(csv_file)

# Step 4: Filter the DataFrame to include only rows where 'Topic' is in the given set of topics.
filtered_df = df[df['Topic'].isin(topics)]

# Step 5: Create the output folder if it doesn't already exist.
os.makedirs(output_folder, exist_ok=True)

# Step 6: Iterate over each row in the filtered DataFrame.
for index, row in filtered_df.iterrows():
    # Extract the base filename without the .txt extension.
    filename_base = row['Filename'].replace('.txt', '')
    
    # Generate a unique name for the paragraph using Original_Doc_Index and the row's index in the DataFrame.
    generated_name = f"para_{row['Original_Doc_Index']}_{index}"
    
    # Construct the new filename: base + '_' + generated_name + '.txt'.
    new_filename = f"{filename_base}_{generated_name}.txt"
    
    # Create the full path for the output file in the common folder.
    file_path = os.path.join(output_folder, new_filename)
    
    # Write the paragraph content to the new text file.
    # The paragraph is written as-is, including any special characters or formatting.
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(row['paragraph'])

# Step 7: Print a completion message with the number of files created (optional, for user feedback).
print(f"Processing complete. {len(filtered_df)} text files have been saved in the '{output_folder}' folder.")