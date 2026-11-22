import numpy as np

from upi_ids.data.generator import FEATURE_COLUMNS, RAW_COLUMNS, UPITransactionGenerator, haversine_km
from upi_ids.features.extractor import FeatureExtractor
from upi_ids.preprocessing.pipeline import Preprocessor, anonymize_value


def test_generator_exact_imbalance():
    df = UPITransactionGenerator(n_users=200, seed=1).generate(2000, fraud_count=20)
    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 2000 and df["is_fraud"].sum() == 20
    assert df["timestamp"].is_monotonic_increasing
    assert df["txn_id"].is_unique


def test_haversine_known_distance():
    d = haversine_km(28.6139, 77.2090, 19.0760, 72.8777)  # Delhi -> Mumbai
    assert 1100 < d < 1200


def test_preprocessing_anonymises_and_dedups():
    df = UPITransactionGenerator(n_users=50, seed=2).generate(200, fraud_count=2)
    dup = df.iloc[[0]]
    out = Preprocessor().transform(__import__("pandas").concat([df, dup]))
    assert len(out) == 200
    assert out["sender_vpa"].iloc[0] == anonymize_value(df["sender_vpa"].iloc[0])
    assert "merchant_risk" in out and out["merchant_risk"].between(0, 1).all()


def test_feature_extractor_shape_and_no_nans():
    df = Preprocessor().transform(UPITransactionGenerator(n_users=50, seed=3).generate(500, fraud_count=5))
    X = FeatureExtractor().transform(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 500 and not X.isna().any().any()
    assert (X["Txn_Count_1Hr"] >= 0).all()  # prior events of the user in the last hour
    assert np.isfinite(X["z_amt"]).all()
