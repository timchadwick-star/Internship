import pandas as pd
from pathlib import Path


def load_data():
    data_dir = Path("data/Raw data")
    data_by_id = {}

    for file_path in data_dir.glob("*.txt"):

        file_id = file_path.stem

        # Find the line where the actual data begins
        with open(file_path, "r") as f:
            lines = f.readlines()

        start_line = None

        for i, line in enumerate(lines):
            if "Time (seconds) and Data Channels" in line:
                start_line = i + 1
                break

        if start_line is None:
            print(f"Could not find data header in {file_path}")

        # Read only the data after the header
        data = pd.read_csv(
            file_path,
            skiprows=start_line,
            sep=r"\s+",
            header=None
            )
        
        data.columns = ["Time", "X", "Y", "Z", "Sound"]

        data_by_id[file_id] = data

    return(data_by_id)
    
    