import sys
sys.path.insert(1, '..')

from gerumo.data.dataset import (
    load_dataset, split_dataset, save_dataset, describe_dataset
)
import argparse
from glob import glob
from os import path, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split extracted dataset.")
    parser.add_argument("-i", "--dataset", type=str, required=True, 
                    help="Folder with events and telescopes parquet datasets.")
    parser.add_argument("-o", "--output", type=str, required=True, 
                    help="Output folder.")
    parser.add_argument("-s", "--split", type=float, default=0.1,
                    help="Validation ratio for split data.")
    args = parser.parse_args()

    assert 0 < args.split and args.split < 1, "split not in (0, 1) range"  
    events_folder = path.join(args.dataset, "events")
    telescopes_folder = path.join(args.dataset, "telescopes")
    dataset = load_dataset(events_folder, telescopes_folder)
    train_dataset, val_dataset = split_dataset(dataset, args.split)
    makedirs(args.output,  exist_ok=True)
    save_dataset(train_dataset, args.output, "train")
    save_dataset(val_dataset, args.output, "validation")
    print("\ntrain_dataset:")
    describe_dataset(train_dataset)
    print("\nval_dataset:")
    describe_dataset(val_dataset)
