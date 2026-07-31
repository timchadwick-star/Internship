from pathlib import Path
import pandas as pd

data_dir = Path("data/Raw data")
index_list = []

for file in data_dir.glob("*.txt"):
    file_id = file.stem

    health_label, condition = file_id.split("_")

    index_list.append(
        {"File ID": file_id,
        "Health label": health_label,
        "Condition": condition,
        "Modality": 'multimodal',
        "File path": str(file)}
        )

data_index = pd.DataFrame(index_list)
print(data_index)
data_index.to_csv("data_index.csv", index = False)