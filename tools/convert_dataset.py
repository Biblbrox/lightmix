import argparse
from os import mkdir
import io
from os.path import exists, isfile
from pathlib import Path

import polars as pl
from tqdm import tqdm
from PIL import Image


def process_imagenet1k(bytes):
    img = Image.open(io.BytesIO(bytes))
    img = img.resize((256, 256))
    img = img.crop((16, 16, 256 - 16, 256 - 16))
    return img.tobytes()


def process_cifar100(bytes):
    img = Image.open(io.BytesIO(bytes))
    return img.tobytes()


def process_mnist(bytes):
    img = Image.open(io.BytesIO(bytes))
    return img.tobytes()


def write_arrow(parquet_path: Path, out_path: Path, dataset):
    assert dataset in ["imagenet1k", "cifar100", "mnist"]
    batch_size = 2048
    df = pl.scan_parquet(parquet_path)
    # print(df.schema)
    arrow_path = out_path.joinpath(f"{parquet_path.stem}.arrow")

    if dataset == "imagenet1k":
        df = df.unnest("image").drop("path").rename({"bytes": "image"})
        process = process_imagenet1k
    elif dataset == "cifar100":
        df = df.unnest("img").drop("path").rename({"bytes": "image"})
        process = process_cifar100
    elif dataset == "mnist":
        df = df.unnest("image").drop("path").rename({"bytes": "image"})
        process = process_mnist

    df.with_columns(
        pl.col("image").map_elements(
            lambda bytes: process(bytes), return_dtype=pl.Binary
        )
    ).sink_ipc(arrow_path, record_batch_size=batch_size, compression="lz4")
    # print(df.schema)


def convert(in_path: Path, out_path: Path, dataset):
    parquet_train = in_path.glob("**/train-*.parquet")
    parquet_test = in_path.glob("**/test-*.parquet")
    parquet_val = in_path.glob("**/val-*.parquet")

    if exists(out_path) and isfile(out_path):
        print(f"Path {out_path} already exists and it's a file")
    elif not exists(out_path):
        mkdir(out_path)

    print("Writing train dataset:")
    for f in tqdm(parquet_train):
        write_arrow(f, out_path, dataset)

    print("Writing test dataset:")
    for f in tqdm(parquet_test):
        write_arrow(f, out_path, dataset)

    print("Writing val dataset:")
    for f in tqdm(parquet_val):
        write_arrow(f, out_path, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset converter tool")
    parser.add_argument(
        "-i", dest="i", help="Path to parquet dataset", type=str, required=True
    )
    parser.add_argument(
        "-o", dest="o", help="Path to output arrow dataset", type=str, required=True
    )
    parser.add_argument(
        "-d",
        dest="d",
        help="Dataset type",
        type=str,
        required=True,
        choices=["imagenet1k", "cifar100", "mnist"],
    )
    args = parser.parse_args()

    convert(Path(args.i), Path(args.o), args.d)
