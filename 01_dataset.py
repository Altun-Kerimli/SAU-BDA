
import kagglehub
import csv
from pathlib import Path

# Download the first candidate dataset
path = kagglehub.dataset_download(
    "usdot/flight-delays"
)

print("Dataset location:", path)

# Find all downloaded files
files = [f for f in Path(path).rglob("*") if f.is_file()]

print("\n=== DATASET FILES ===")
for file in files:
    size_mb = file.stat().st_size / (1024 ** 2)
    print(f"{file.name}: {size_mb:.2f} MB")

# Inspect CSV files
for file in files:
    if file.suffix.lower() != ".csv":
        continue

    print(f"\n=== {file.name} ===")

    with open(file, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)

        columns = next(reader, [])
        print("Number of columns:", len(columns))
        print("Columns:", columns)

        print("\nFirst 3 records:")
        for _, row in zip(range(3), reader):
            print(row)
