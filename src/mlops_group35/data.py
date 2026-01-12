from pathlib import Path
import pandas as pd

import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_path}")

        # Read and combine
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(df.shape)

        self.data = pd.concat(dfs, ignore_index=True)
        print(self.data.shape)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.data[index]
        #return self.data.iloc[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / "combined.csv"
        self.data.to_csv(output_file, index=False)


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)
    print(dataset.data.head(10))


if __name__ == "__main__":
    print("Preprocessing data main...")
    typer.run(preprocess)
    #preprocess(Path("data/raw"), Path("data/preprocessed"))
