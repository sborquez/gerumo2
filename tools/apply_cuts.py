import sys; sys.path.insert(1, '..')  # noqa
from os import path, makedirs, rename
import argparse
from gerumo.data.dataset import (
    load_dataset, save_dataset, describe_dataset, apply_cut_to_dataset
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply cuts to dataset.')
    parser.add_argument('-i', '--dataset', type=str, required=True,
                        help='Folder with events and telescopes parquet datasets.')  # noqa
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output folder.')
    parser.add_argument('-T', '--threshold', type=float, default=1000,
                        help='Theshold for the cut.')
    parser.add_argument('-C', '--cut', type=str, default='hillas_intensity', choices=('hillas_intensity', 'true_energy'),
                        help='Column to cut.')
    args = parser.parse_args()

    events_folder = path.join(args.dataset, 'events')
    telescopes_folder = path.join(args.dataset, 'telescopes')
    dataset = load_dataset(events_folder, telescopes_folder)
    cut_dataset = apply_cut_to_dataset(dataset, cut=args.cut, threshold=args.threshold)
    makedirs(args.output, exist_ok=False)
    makedirs(path.join(args.output, 'events'), exist_ok=True)
    makedirs(path.join(args.output, 'telescopes'), exist_ok=True)
    save_dataset(cut_dataset, args.output)
    rename(path.join(args.output, 'events.parquet'), path.join(args.output, 'events', 'events.parquet'))
    rename(path.join(args.output, 'telescopes.parquet'), path.join(args.output, 'telescopes', 'telescopes.parquet'))
    print('\ncut_dataset:')
    describe_dataset(cut_dataset)