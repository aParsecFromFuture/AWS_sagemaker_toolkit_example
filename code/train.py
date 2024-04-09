import os
import sys
import joblib
import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


feature_columns_names = [
    "Loan_ID",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicationIncome",
    "CoapplicationIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

label_column = "Loan_Status"

columns_dtype = {
    "Loan_ID": "object",
    "ApplicationIncome": "float64",
    "CoApplicationIncome": "float64",
    "LoanAmount": "float64",
    "Loan_Amount_Term": "float64",
    "Gender": "category",
    "Married": "category",
    "Dependents": "category",
    "Education": "category",
    "Credit_History": "category",
    "Property_Area": "category",
    "Loan_Status": "category",
}

if __name__ == '__main__':
    # Define command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100)  # Number of trees in randomforest (default: 100)
    parser.add_argument('--criterion', type=str, default="gini")  # The function to measure the quality of a split (default: "gini")
    parser.add_argument('--max_depth', type=int, default=10)  # The maximum depth of the tree (default: 10)
    
    # SageMaker environment variables for I/O
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Construct input file paths
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    
    # Check if any input files are found
    if len(input_files) == 0:
        raise ValueError(("There are no files in {}.\n" +
                          "This usually indicates that the channel ({}) was incorrectly specified, \n" +
                          "the data specification in S3 was incorrectly specified, or the specified role\n" +
                          "does not have permission to access the data.").format(args.train, "train"))
    
    # Read and concatenate the raw data
    raw_data = [pd.read_csv(
        file,
        header=None,
        names=[label_column] + feature_columns_names,
        dtype=columns_dtype) for file in input_files
    ]

    concat_data = pd.concat(raw_data)

    # Split features and labels
    X = concat_data.iloc[:, 1:]  # Features
    y = concat_data.iloc[:, 0]   # Labels

    # Train-Validation splitting
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5)

    # Define preprocessing pipeline
    preprocessor = make_column_transformer(
        (Pipeline([
            ("imputer", SimpleImputer(strategy="mean")), 
        ]), ["ApplicationIncome", "CoapplicationIncome", "LoanAmount", "Loan_Amount_Term"]),
        (Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")), 
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=1024)),
        ]), ["Gender", "Married", "Dependents", "Education"]),
        (Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")), 
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), ["Property_Area"]),
    )
    
    # Define estimator
    estimator = RandomForestClassifier(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=args.max_depth,
    )

    # Create a pipeline with preprocessor and estimator
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator)
    ])

    # Training and validation
    model.fit(X_train, y_train)
    score = model.score(X_valid, y_valid)

    print("Validation score: {0:.3f}".format(score))
    
    # Save the trained model to the model directory
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    print("Model successfully saved to ", os.path.join(args.model_dir, "model.joblib"))

