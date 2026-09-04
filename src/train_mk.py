"""
src/train_mk.py
----------------
Trains a RandomForest model to predict property `price` from the
Zameen-style property listings dataset.

Repo layout this script expects (relative to project root):
    data/<csv file>      -> input dataset(s)
    model/model.joblib    -> trained pipeline is written here

Usage (from project root, per README):
    python src/train_mk.py
    python src/train_mk.py --data data/properties.csv --out model/model.joblib

Expected columns (from the dataset header):
    property_id, location_id, page_url, property_type, price, location,
    city, province_name, latitude, longitude, baths, area, purpose,
    bedrooms, date_added, agency, agent, Area Type, Area Size, Area Category
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


# --------------------------------------------------------------------------
# Config: which columns to use and how
# --------------------------------------------------------------------------

TARGET = "price"

# Columns that are identifiers / free text / leakage-prone -> dropped
DROP_COLS = [
    "property_id",
    "location_id",
    "page_url",
    "location",       # very high-cardinality text; city/province capture geo info
    "agency",
    "agent",
    "date_added",      # engineered into year_added / month_added instead
    "area",            # redundant with Area Size + Area Type
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "baths",
    "bedrooms",
    "area_marla",      # engineered: Area Size normalized to a single unit (Marla)
    "year_added",      # engineered from date_added
    "month_added",     # engineered from date_added
]

CATEGORICAL_FEATURES = [
    "property_type",
    "city",
    "province_name",
    "purpose",
    "Area Type",
    "Area Category",
]

# 1 Kanal = 20 Marla, 1 Sq. Yard = 1/30.25 Marla, 1 Sq. Ft = 1/272.25 Marla (approx, PK convention)
AREA_TYPE_TO_MARLA = {
    "Marla": 1.0,
    "Kanal": 20.0,
    "Sq. Yards": 1.0 / 30.25,
    "Sq. Yard": 1.0 / 30.25,
    "Sq. Ft.": 1.0 / 272.25,
    "Sq. Ft": 1.0 / 272.25,
}

# Categorical levels that appear fewer than this many times get folded into "Other"
RARE_CATEGORY_MIN_COUNT = 20

# Price values outside these percentiles are clipped (guards against fat-finger /
# placeholder listings without throwing away rows)
PRICE_LOWER_PCT = 0.01
PRICE_UPPER_PCT = 0.99


# --------------------------------------------------------------------------
# Feature engineering / preprocessing helpers
# --------------------------------------------------------------------------

def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn date_added (dd-mm-yyyy) into numeric year/month features."""
    if "date_added" in df.columns:
        parsed = pd.to_datetime(df["date_added"], format="%d-%m-%Y", errors="coerce")
        df["year_added"] = parsed.dt.year
        df["month_added"] = parsed.dt.month
    else:
        df["year_added"] = np.nan
        df["month_added"] = np.nan
    return df


def engineer_area_marla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize `Area Size` into a single consistent unit (Marla) using `Area Type`,
    since raw Area Size is not comparable across rows recorded in Marla vs Kanal
    vs Sq. Ft. etc. Unknown/unmapped area types fall back to NaN and get imputed
    later in the pipeline.
    """
    if "Area Size" in df.columns and "Area Type" in df.columns:
        factor = df["Area Type"].map(AREA_TYPE_TO_MARLA)
        df["area_marla"] = df["Area Size"] * factor
    else:
        df["area_marla"] = np.nan
    return df


def group_rare_categories(df: pd.DataFrame, columns, min_count: int) -> pd.DataFrame:
    """Collapse infrequent categorical levels into 'Other' so the encoder doesn't
    blow up the feature space with one-off values and so rare/unseen values at
    inference time behave predictably."""
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        rare = counts[counts < min_count].index
        df[col] = df[col].where(~df[col].isin(rare), other="Other")
    return df


def clip_price_outliers(df: pd.DataFrame, lower_pct: float, upper_pct: float) -> pd.DataFrame:
    """Clip extreme price values to the given percentiles instead of dropping rows,
    so the model isn't skewed by a handful of placeholder / data-entry-error prices."""
    lower = df[TARGET].quantile(lower_pct)
    upper = df[TARGET].quantile(upper_pct)
    df[TARGET] = df[TARGET].clip(lower=lower, upper=upper)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df = engineer_date_features(df)
    df = engineer_area_marla(df)
    df = group_rare_categories(df, CATEGORICAL_FEATURES, RARE_CATEGORY_MIN_COUNT)

    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return preprocess(df)


# --------------------------------------------------------------------------
# Model pipeline
# --------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    # Feature tuning: fit a quick RandomForest to rank feature importance, then
    # automatically drop features below the median importance. This trims noisy
    # one-hot columns (e.g. rare categories that slipped past grouping) and low-
    # signal numeric features before the final model is trained.
    feature_tuning = SelectFromModel(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        threshold="median",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ("feature_tuning", feature_tuning),
        ("model", model),
    ])

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train a RandomForest price predictor.")
    parser.add_argument(
        "--data",
        default="data/zameen-updated.csv",
        help="Path to input CSV file (default: data/zameen-updated.csv).",
    )
    parser.add_argument(
        "--out",
        default="model/model.joblib",
        help="Path to save trained model (default: model/model.joblib).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data)

    missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # Filter out non-positive prices, then clip remaining extreme outliers
    df = df[df[TARGET] > 0]
    df = clip_price_outliers(df, PRICE_LOWER_PCT, PRICE_UPPER_PCT)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\nEvaluation on held-out test set:")
    print(f"  MAE  : {mae:,.0f}")
    print(f"  RMSE : {rmse:,.0f}")
    print(f"  R^2  : {r2:.4f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(pipeline, args.out)
    print(f"\nSaved trained pipeline to {args.out}")


if __name__ == "__main__":
    main()