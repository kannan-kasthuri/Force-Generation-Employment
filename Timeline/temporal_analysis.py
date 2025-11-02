import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

# Function to read document metadata from CSV and compute theme counts per year
def read_document_metadata(file_path):
    """
    Reads document metadata from a CSV file and extracts year and theme.
    Returns a dictionary mapping years to theme counts, excluding 'Outlier' theme.
    """
    year_theme_counts = defaultdict(lambda: defaultdict(int))
    total_documents = 0
    parsing_errors = 0

    try:
        # Read CSV file
        df = pd.read_csv(file_path, encoding='utf-8')
        print("Reading file:", file_path)
        
        # Filter out rows where Theme is 'Outlier'
        df = df[df['Theme'] != 'Outlier']
        
        # Group by Year and Theme to count occurrences
        theme_counts = df.groupby(['Year', 'Theme']).size().unstack(fill_value=0)
        
        # Populate year_theme_counts
        for year in theme_counts.index:
            for theme in theme_counts.columns:
                year_theme_counts[year][theme] += theme_counts.loc[year, theme]
                total_documents += theme_counts.loc[year, theme]
        
        print(f"Total documents processed (excluding Outlier): {total_documents}")
        print(f"Parsing errors encountered: {parsing_errors}")
        
        # Print documents per year
        print("\nDocuments per year:")
        for year in sorted(year_theme_counts.keys()):
            year_total = sum(year_theme_counts[year].values())
            print(f"Year {year}: {year_total} documents")
        
        # Print theme counts per year
        print("\nTheme counts per year:")
        themes = sorted(set(theme for year in year_theme_counts for theme in year_theme_counts[year]))
        for year in sorted(year_theme_counts.keys()):
            print(f"Year {year}:")
            for theme in themes:
                count = year_theme_counts[year].get(theme, 0)
                print(f"  Theme {theme}: {count} documents")
    
    except FileNotFoundError:
        print(f"File {file_path} not found. Using placeholder data.")
        # Placeholder data for demonstration
        themes = [
            "Operational Readiness and Reserve Mobilization",
            "Strategic Command, Technology, and Multi-Domain Warfare"
        ]
        years = list(range(2014, 2025))
        for year in years:
            for theme in themes:
                year_theme_counts[year][theme] = 100  # Example count
        total_documents = len(years) * len(themes) * 100
        print("Warning: Using placeholder data with uniform distribution across years.")
        print(f"Total documents in placeholder: {total_documents}")

    return year_theme_counts

# Read document metadata
USAWC = os.getenv('USAWC')
INPUT_FOLDER = f"{USAWC}\\Desktop\\Data\\USAWC_EDU\\Research\\SQRL\\Drafts\\Manuscripts\\BERTopic_Analysis\\Figures\\Timeline"
file_path = f"{INPUT_FOLDER}\\Theme_With_Year_Info.csv"
year_theme_counts = read_document_metadata(file_path)

# Prepare data for plotting
years = sorted([y for y in range(2014, 2025) if y in year_theme_counts])
themes = sorted(set(theme for year in year_theme_counts for theme in year_theme_counts[year]))
proportions = {theme: [] for theme in themes}

# Calculate proportions for each year
print("\nProportions per year:")
for year in years:
    total_docs = sum(year_theme_counts[year].values())
    print(f"Year {year} (Total: {total_docs} documents):")
    if total_docs == 0:
        for theme in themes:
            proportions[theme].append(0.0)
            print(f"  Theme {theme}: 0.0")
    else:
        for theme in themes:
            count = year_theme_counts[year].get(theme, 0)
            prop = count / total_docs if total_docs > 0 else 0.0
            proportions[theme].append(prop)
            print(f"  Theme {theme}: {prop:.4f}")
    # Verify proportions sum
    year_sum = sum(proportions[theme][-1] for theme in themes)
    print(f"  Sum of proportions: {year_sum:.4f}")

# Plot stacked area chart
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(themes)))
plt.stackplot(years, [proportions[theme] for theme in themes],
              labels=themes, colors=colors, alpha=0.8)

# Customize plot
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.title("Temporal Trend of Themes (2014–2024)")
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save the plot
plt.savefig("theme_temporal_distribution.png", format="png", bbox_inches="tight")
plt.close()

# Note: To include in LaTeX, use \includegraphics{theme_temporal_distribution.pdf}
# Caption: Temporal evolution of theme proportions from 2014–2024, excluding Outlier theme.