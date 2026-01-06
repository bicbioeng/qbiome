import qbiome
from qbiome.data_formatter import DataFormatter
from qbiome.quantizer import Quantizer
from qbiome.qnet_orchestrator import QnetOrchestrator
from qbiome.forecaster import Forecaster
from qbiome.hypothesis import Hypothesis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
# from qbiome.qutil import smooth


# File path to your merged CSV file
file_path = "example_data/merged_arg_gene_abundance.csv"  # update with actual path if needed

# Read the CSV into a DataFrame

merged_df = pd.read_csv(file_path,keep_default_na=False)

# Preview the first few rows
print(merged_df.head())

# Compute threshold (50% of total rows)
threshold = len(merged_df) * 0.01

# Identify numeric columns (excluding CITY, sample_id, etc.)
numeric_cols = merged_df.select_dtypes(include=[np.number])

# Find columns where 0 appears in ≥ 50% of the rows
cols_to_drop = [col for col in numeric_cols.columns if (merged_df[col] == 0).sum() >= threshold]

# Drop the columns from the original dataframe
filtered_df = merged_df.drop(columns=cols_to_drop)

# Show how many were removed and preview
print(f"Removed {len(cols_to_drop)} ")
print(f"New shape: {filtered_df.shape}") 
filtered_df.head()


merged_df = filtered_df


import pandas as pd

# Assuming 'merged_df' is your original DataFrame (from first image)
long_df = merged_df.melt(
    id_vars=['sample_id', 'CITY', 'relative_week'],
    var_name='variable',
    value_name='value'
)

# Optionally rename columns to match second image format
long_df.rename(columns={
    'CITY': 'subject_id',
    'relative_week': 'week'
}, inplace=True)

# Reorder columns
long_df = long_df[['sample_id', 'subject_id', 'variable', 'value', 'week']]

# Preview the transformed DataFrame
print(long_df.head())

data= long_df

print("Final trained data ,",data.shape)
print("unique ARG,",len(data.variable.unique()))

print(data.value.min(), data.value.max())
data['value'] = np.log2(data['value'] + 1)
base_tax = "Time_Series_ARG_50percent_arg_removed"

print("Quantizing the values ...") 
quantizer = Quantizer(num_levels=26)
qnet_orchestrator = QnetOrchestrator(quantizer)
quantized = quantizer.quantize_df(data)


features, label_matrix = quantizer.get_qnet_inputs(quantized)

print("Quantization is done...") 

TAXA = list(data.variable.value_counts().index.values)

print("Starting to train the model ...") 
qnet_orchestrator.train_qnet(
    features, label_matrix, alpha=0.3, min_samples_split=2, out_fname=None
)

print("Saving the model...") 
qnet_orchestrator.save_qnet("example_qnet_"+base_tax+".pkl",GZIP=True)

print("Model is saved...")


