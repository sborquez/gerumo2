import sys; sys.path.insert(1, '..')  # noqa
import argparse
from glob import glob
from os import path
import time
from gerumo.data.dataset import (
    generate_dataset_multiprocess, load_dataset, describe_dataset
)

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare reference dataset for generators.')  # noqa
    gp = parser.add_mutually_exclusive_group(required=True)
    gp.add_argument('-i', '--folder', type=str, default=None,
                    help='Folder containing hdf5 files.')
    gp.add_argument('-I', '--files', nargs='*', type=str, default=[],
                    help='List of hdf5 files, if is not empty, ignore folder argument.')  # noqa
    gp.add_argument('-f', '--file', type=str, default=None,
                    help='File with list of hdf5 files, if is not None, ignore folder argument.')  # noqa
    parser.add_argument('-p', '--processes', type=int, required=True,
                        help='Ouput folder.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Ouput folder.')
    parser.add_argument('-w', '--overwrite', dest='overwrite', action='store_true')  # noqa
    args = parser.parse_args()

    # Load files from a txt list of h5
    if args.file is not None:
        with open(args.file) as f:
            files = [file_.strip() for file_ in f.readlines()]
    # Load files form folder
    elif args.folder is not None:
        files = glob(path.join(args.folder, '*.h5'))
        files = [path.abspath(file) for file in files]
    # Load files from cmd args
    else:
        files = args.files

    # Process the list of files
    if len(files) > 0:
        t = time.process_time()
        events_folder, telescopes_folder = generate_dataset_multiprocess(
            file_paths=files, output_folder=args.output,
            processes=args.processes, append=not args.overwrite
        )
        logging.info(f'Elapsed time {time.process_time() - t}')
    else:
        raise ValueError(f'folder or files not set correctly. (len(files)={len(files)})')  # noqa

    dataset = load_dataset(events_folder, telescopes_folder)

    print('Dataset')
    describe_dataset(dataset)
