import os

from tests import _PATH_DATA
import pandas as pd

NUM_ROWS = 606
NUM_COLS = 23
CSV_PATH = os.path.join(_PATH_DATA, "processed", "combined.csv")
DF = pd.read_csv(CSV_PATH)
# use df = DF.copy(deep=True) if you are going to modify the data in any ways


def test_dataset():
    """Test the MyDataset class."""
    df = DF
    assert df is not None
    assert not df.empty, "Dataset should not be empty"

    assert len(df.columns) == NUM_COLS, f"Dataset should have {NUM_COLS} columns"

    print(len(df))
    assert len(df) >= NUM_ROWS, f"Dataset should have at least {NUM_ROWS} rows"


REQUIRED_COLUMNS = {
    "scandir_id",
    "site",
    "gender",
    "age",
    "handedness",
    "dx",
    "secondary_dx",
    "adhd_measure",
    "adhd_index",
    "inattentive",
    "hyper_impulsive",
    "iq_measure",
    "verbal_iq",
    "performance_iq",
    "full2_iq",
    "full4_iq",
    "med_status",
    "qc_rest_1",
    "qc_rest_2",
    "qc_rest_3",
    "qc_rest_4",
    "qc_anatomical_1",
    "qc_anatomical_2",
}


def test_required_columns():
    df = DF
    print(df.columns)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_all_missing_columns():
    df = DF
    all_missing = df.columns[df.isna().all()]
    assert len(all_missing) == 0, f"Columns with only NaNs: {list(all_missing)}"
