"""Data schema and validation for the fraud detection pipeline."""

import logging
import sys

import pandas as pd
import pandera.pandas as pa

log = logging.getLogger("fraud.validate")

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", *V_COLS, "Amount"]

V_RANGES = {
    "V1": (-68.176853, 14.208551),
    "V2": (-91.670419, 41.012420),
    "V3": (-42.293693, 17.995267),
    "V4": (-9.998705, 21.354352),
    "V5": (-57.537811, 50.191579),
    "V6": (-35.898467, 32.267259),
    "V7": (-59.604143, 52.724170),
    "V8": (-91.861504, 38.651994),
    "V9": (-19.239879, 21.400807),
    "V10": (-34.254942, 33.411816),
    "V11": (-7.263657, 15.232675),
    "V12": (-23.990136, 13.154813),
    "V13": (-7.864059, 6.641187),
    "V14": (-25.162544, 16.474984),
    "V15": (-6.555636, 7.841206),
    "V16": (-18.175531, 10.144206),
    "V17": (-32.046064, 16.136791),
    "V18": (-12.406709, 7.949032),
    "V19": (-9.701901, 7.716716),
    "V20": (-35.290439, 51.872795),
    "V21": (-47.237026, 39.609483),
    "V22": (-15.220390, 14.790337),
    "V23": (-57.569871, 31.765077),
    "V24": (-4.208525, 5.394764),
    "V25": (-13.858394, 11.082586),
    "V26": (-3.828930, 4.741725),
    "V27": (-29.509295, 19.096017),
    "V28": (-20.822636, 42.959549),
}

AMOUNT_MAX = 3000.0


_feature_cols = {
    "Time": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=False,
        coerce=True,
    ),
    "Amount": pa.Column(
        float,
        [
            pa.Check.ge(0),
            pa.Check.le(AMOUNT_MAX),
        ],
        nullable=False,
        coerce=True,
    ),
    **{
        col: pa.Column(
            float,
            pa.Check.in_range(min_value, max_value),
            nullable=False,
            coerce=True,
        )
        for col, (min_value, max_value) in V_RANGES.items()
    },
}


feature_schema = pa.DataFrameSchema(
    _feature_cols,
    strict=True,
)


training_schema = pa.DataFrameSchema(
    {
        **_feature_cols,
        "Class": pa.Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        ),
    },
    strict=True,
)


def validate(
    df: pd.DataFrame,
    with_target: bool = True,
) -> pd.DataFrame:
    """Validate a dataframe against the expected schema."""
    schema = training_schema if with_target else feature_schema

    try:
        return schema.validate(df, lazy=True)

    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases

        log.error(
            "DATA VALIDATION FAILED: %d problems",
            len(cases),
        )

        if "column" in cases.columns:
            column_counts = (
                cases["column"]
                .fillna("DATAFRAME")
                .value_counts()
            )

            for column, count in column_counts.items():
                log.error(
                    "Column %s has %d validation problem(s)",
                    column,
                    count,
                )

        log.error(
            "Failure details:\n%s",
            cases.head(20),
        )

        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/creditcard.csv"
    )

    try:
        validate(pd.read_csv(path))

    except pa.errors.SchemaErrors:
        sys.exit(1)

    log.info("data OK: %s", path)