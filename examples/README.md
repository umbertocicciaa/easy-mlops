# Easy MLOps Examples

This directory contains example data and usage scenarios for Easy MLOps.

## Sample Data

### `sample_data.csv`

A simple loan approval dataset with the following features:
- `age`: Age of the applicant
- `income`: Annual income
- `credit_score`: Credit score
- `loan_amount`: Requested loan amount
- `approved`: Whether the loan was approved (target variable)

## Usage Examples

### 1. Basic Training

Train a model with default settings:

```bash
easy-mlops train examples/sample_data.csv --target approved
```

### 2. Training with Custom Configuration

First, create a custom configuration:

```bash
easy-mlops init -o examples/custom-config.yaml
```

Edit the configuration file as needed, then train:

```bash
easy-mlops train examples/sample_data.csv --target approved -c examples/custom-config.yaml
```

### 3. Making Predictions

After training, use the model to make predictions on new data:

```bash
# Assuming the model was deployed to models/deployment_TIMESTAMP
easy-mlops predict examples/sample_data.csv models/deployment_*/
```

### 4. Monitoring

Check the status of your deployed model:

```bash
easy-mlops status models/deployment_*/
```

Generate an observability report:

```bash
easy-mlops observe models/deployment_*/
```

### 5. Using the Prediction Endpoint

After deployment with endpoint creation enabled, you can use the generated script:

```bash
python models/deployment_*/predict.py examples/sample_data.csv
```

## Complete Workflow Example

```bash
# Step 1: Initialize project
easy-mlops init -o my-config.yaml

# Step 2: Train model
easy-mlops train examples/sample_data.csv --target approved -c my-config.yaml

# Step 3: Check status
easy-mlops status models/deployment_*/

# Step 4: Make predictions
easy-mlops predict examples/sample_data.csv models/deployment_*/ -o predictions.json

# Step 5: Monitor performance
easy-mlops observe models/deployment_*/
```

## Creating Your Own Dataset

Easy MLOps supports CSV, JSON, and Parquet formats. Your data should:

1. Have a clear target column for training
2. Include both numerical and/or categorical features
3. Be in one of the supported formats

Example CSV structure:
```csv
feature1,feature2,feature3,target
value1,value2,value3,label1
value4,value5,value6,label2
```

Example JSON structure:
```json
[
  {"feature1": "value1", "feature2": "value2", "target": "label1"},
  {"feature1": "value3", "feature2": "value4", "target": "label2"}
]
```
