import argparse
import io
from os import mkdir
from os.path import exists, isfile
from pathlib import Path

import polars as pl
from PIL import Image
from PIL.Image import Resampling

DATASET_SPECS = {
    "imagenet1k": {
        "labelcol": "label",
        "imagecol": "image",
        "size": (256, 256),
        "crop": (16, 16, 240, 240),
        "channels": 3,
    },
    "tinyimagenet": {
        "labelcol": "label",
        "imagecol": "image",
        "size": (64, 64),
        "crop": None,
        "channels": 3,
    },  # Broken
    "cifar100": {
        "labelcol": "fine_label",
        "imagecol": "img",
        "size": None,
        "crop": None,
        "channels": 3,
    },
    "cifar10": {
        "labelcol": "label",
        "imagecol": "img",
        "size": None,
        "crop": None,
        "channels": 3,
    },
    "mnist": {
        "labelcol": "label",
        "imagecol": "image",
        "size": None,
        "crop": None,
        "channels": 1,
    },
    "fashionmnist": {
        "labelcol": "label",
        "imagecol": "image",
        "size": None,
        "crop": None,
        "channels": 1,
    },
    "food101": {
        "labelcol": "label",
        "imagecol": "image",
        "size": (96, 96),
        "crop": None,
        "channels": 3,
    },
}


def decode(raw: bytes, spec: dict) -> bytes:
    img = Image.open(io.BytesIO(raw))
    if spec["size"]:
        img = img.resize(spec["size"], Resampling.BICUBIC)
    if spec["crop"]:
        img = img.crop(spec["crop"])
    if spec["channels"] == 3 and img.mode != "RGB":
        img = img.convert("RGB")
    elif spec["channels"] == 1 and img.mode != "L":
        img = img.convert("L")

    raw_bytes = img.tobytes()

    expected = img.width * img.height * len(img.getbands())
    if len(raw_bytes) != expected:
        raise ValueError(f"Corrupted image: got {len(raw_bytes)}, expected {expected}")

    return raw_bytes


def write_arrow(parquet_path: Path, out_path: Path, dataset):
    spec = DATASET_SPECS[dataset]
    batch_size = 128
    arrow_path = out_path.joinpath(f"{parquet_path.stem}.arrow")

    image_col: str = DATASET_SPECS[dataset]["imagecol"]
    df = pl.scan_parquet(parquet_path)
    try:
        schema = df.collect_schema()
    except pl.exceptions.ComputeError:
        print(f"Unable to find files with glob: {parquet_path}. Skipping")
        return
    df = df.unnest(image_col).drop("path").rename({"bytes": "image"})

    print(schema)
    # It would be nice to have a progress bar
    df = df.with_columns(
        pl.col("image").map_elements(lambda bytes: decode(bytes, spec), return_dtype=pl.Binary)
    )

    df.with_columns(
        pl.col(["image", DATASET_SPECS[dataset]["labelcol"]]).shuffle(seed=42)
    ).sink_ipc(arrow_path, record_batch_size=batch_size)


def convert(in_path: Path, out_path: Path, dataset):
    parquet_train = in_path.joinpath("**/train-*.parquet")
    parquet_test = in_path.joinpath("**/test-*.parquet")
    parquet_val = in_path.joinpath("**/val*.parquet")

    if exists(out_path) and isfile(out_path):
        print(f"Path {out_path} already exists and it's a file")
    elif not exists(out_path):
        mkdir(out_path)

    print("Writing train dataset:")
    write_arrow(parquet_train, out_path, dataset)

    print("Writing test dataset:")
    write_arrow(parquet_test, out_path, dataset)

    print("Writing val dataset:")
    write_arrow(parquet_val, out_path, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset converter tool")
    parser.add_argument("-i", dest="i", help="Path to parquet dataset", type=str, required=True)
    parser.add_argument(
        "-o", dest="o", help="Path to output arrow dataset", type=str, required=True
    )
    parser.add_argument(
        "-d",
        dest="d",
        help="Dataset type",
        type=str,
        required=True,
        choices=DATASET_SPECS.keys(),
    )
    args = parser.parse_args()

    convert(Path(args.i), Path(args.o), args.d)
