"""
src/train_mk.py
---------------
Trains a RandomForest model to predict property `price` from the
Zameen-style property listings dataset.

Run from the project root:

    python src/train_mk.py

Or specify your own dataset/output:

    python src/train_mk.py --data data/properties.csv --out model/model.joblib
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TARGET = "price"

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "baths",
    "bedrooms",
    "area_marla",
    "year_added",
    "month_added",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "city",
    "province_name",
    "purpose",
    "Area Type",
    "Area Category",
]

# Convert different area units into Marla.
AREA_TYPE_TO_MARLA = {
    "Marla": 1.0,
    "Kanal": 20.0,
    "Sq. Yards": 1.0 / 30.25,
    "Sq. Yard": 1.0 / 30.25,
    "Sq. Ft.": 1.0 / 272.25,
    "Sq. Ft": 1.0 / 272.25,
}

# Categories occurring fewer than this number of times
# are grouped into "Other".
RARE_CATEGORY_MIN_COUNT = 20

# Percentiles used to clip extreme price values.
PRICE_LOWER_PCT = 0.01
PRICE_UPPER_PCT = 0.99


# --------------------------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------------------------

def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date_added into numeric year and month features."""

    if "date_added" in df.columns:
        parsed = pd.to_datetime(
            df["date_added"],
            format="%d-%m-%Y",
            errors="coerce"
        )

        df["year_added"] = parsed.dt.year
        df["month_added"] = parsed.dt.month

    else:
        df["year_added"] = np.nan
        df["month_added"] = np.nan

    return df


def engineer_area_marla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Area Size into Marla using Area Type.

    This makes area measurements comparable when the dataset contains
    Marla, Kanal, Sq. Yards, or Sq. Ft.
    """

    if "Area Size" in df.columns and "Area Type" in df.columns:

        # Ensure Area Size is numeric.
        area_size = pd.to_numeric(
            df["Area Size"],
            errors="coerce"
        )

        factor = df["Area Type"].map(AREA_TYPE_TO_MARLA)

        df["area_marla"] = area_size * factor

    else:
        df["area_marla"] = np.nan

    return df


def group_rare_categories(
    df: pd.DataFrame,
    columns,
    min_count: int
) -> pd.DataFrame:
    """
    Replace infrequent categorical values with 'Other'.
    """

    for col in columns:

        if col not in df.columns:
            continue

        counts = df[col].value_counts()

        rare_values = counts[
            counts < min_count
        ].index

        df[col] = df[col].where(
            ~df[col].isin(rare_values),
            other="Other"
        )

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Perform all feature engineering and preprocessing."""

    df = df.copy()

    # Remove accidental whitespace from column names.
    df.columns = [c.strip() for c in df.columns]

    # Convert relevant numeric columns.
    numeric_columns = [
        "price",
        "latitude",
        "longitude",
        "baths",
        "bedrooms",
        "Area Size",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Feature engineering.
    df = engineer_date_features(df)
    df = engineer_area_marla(df)

    # Handle rare categorical values.
    df = group_rare_categories(
        df,
        CATEGORICAL_FEATURES,
        RARE_CATEGORY_MIN_COUNT
    )

    return df


def clip_price_outliers(
    df: pd.DataFrame,
    lower_pct: float,
    upper_pct: float
) -> pd.DataFrame:
    """Clip extreme target values instead of removing rows."""

    lower = df[TARGET].quantile(lower_pct)
    upper = df[TARGET].quantile(upper_pct)

    df[TARGET] = df[TARGET].clip(
        lower=lower,
        upper=upper
    )

    return df


# --------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load CSV data and apply preprocessing."""

    print(f"Loading dataset from: {path}")

    df = pd.read_csv(path)

    print(f"Dataset loaded: {len(df)} rows")

    df = preprocess(df)

    return df


# --------------------------------------------------------------------------
# Machine Learning Pipeline
# --------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Build the preprocessing + feature selection + RandomForest pipeline."""

    # Numeric preprocessing.
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            # Required feature scaling step.
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # Categorical preprocessing.
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    # Combine numeric and categorical preprocessing.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                NUMERIC_FEATURES
            ),

            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES
            ),
        ]
    )

    # Feature selection.
    feature_tuning = SelectFromModel(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        threshold="median"
    )

    # Final RandomForest model.
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "feature_tuning",
                feature_tuning
            ),

            (
                "model",
                model
            ),
        ]
    )

    return pipeline


# --------------------------------------------------------------------------
# Main Training Function
# --------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train a RandomForest property price predictor."
    )

    parser.add_argument(
        "--data",
        default="data/zameen-updated.csv",
        help="Path to the input CSV dataset."
    )

    parser.add_argument(
        "--out",
        default="model/model.joblib",
        help="Path where the trained model will be saved."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------------

    df = load_data(args.data)

    # ----------------------------------------------------------------------
    # Validate required columns
    # ----------------------------------------------------------------------

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing expected columns in dataset: "
            f"{missing_columns}"
        )

    # ----------------------------------------------------------------------
    # Clean target
    # ----------------------------------------------------------------------

    # Remove rows where price is missing.
    df = df.dropna(
        subset=[TARGET]
    )

    # Remove invalid/non-positive prices.
    df = df[
        df[TARGET] > 0
    ]

    # Clip extreme prices.
    df = clip_price_outliers(
        df,
        PRICE_LOWER_PCT,
        PRICE_UPPER_PCT
    )

    # ----------------------------------------------------------------------
    # Prepare X and y
    # ----------------------------------------------------------------------

    X = df[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y = df[TARGET]

    # ----------------------------------------------------------------------
    # Train/test split
    # ----------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ----------------------------------------------------------------------
    # Build and train model
    # ----------------------------------------------------------------------

    print("\nTraining RandomForest model...")

    pipeline = build_pipeline()

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # ----------------------------------------------------------------------
    # Evaluate model
    # ----------------------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nEvaluation on test set:")
    print(f"MAE  : {mae:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"R^2  : {r2:.4f}")

    # ----------------------------------------------------------------------
    # Save model
    # ----------------------------------------------------------------------

    output_directory = os.path.dirname(
        args.out
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    joblib.dump(
        pipeline,
        args.out
    )

    print(
        f"\nSaved trained model to: {args.out}"
    )


# --------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    main()
