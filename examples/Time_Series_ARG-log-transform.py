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

print(data.value.min(), data.value.max())
print(len(data.value.unique()))
data['value'] = np.log2(data['value'] + 1)
base_tax = "Time_Series_ARG_log"

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


