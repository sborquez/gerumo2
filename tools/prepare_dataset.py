import sys
sys.path.insert(1, '..')

from gerumo.data import generate_dataset
import argparse
from glob import glob
from os import path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare reference dataset for generators.")
    gp = parser.add_mutually_exclusive_group(required=True)
    gp.add_argument("-i", "--folder", type=str, default=None,
                    help="Folder containing hdf5 files.")
    gp.add_argument("-I", "--files", nargs="*", type=str, default=[],
                    help="List of hdf5 files, if is not empty, ignore folder argument.")
    gp.add_argument("-f", "--file", type=str, default=None,
                    help="File with list of hdf5 files, if is not None, ignore folder argument.")
    parser.add_argument("-o", "--output", type=str, default="./output", 
                    help="Ouput folder.")
    parser.add_argument("-s", "--split", type=float, default=0.1,
                    help="Validation ratio for split data.")
    parser.add_argument("-a", "--append", dest='append_write', action='store_true')       
    args = parser.parse_args()

    # Load files from a txt list of h5
    if args.file is not None:
        with open(args.file) as f:
            files = [l.strip() for l in f.readlines()]
    # Load files form folder
    elif args.folder is not None:
        files = glob(path.join(args.folder, "*.h5"))
        files = [path.abspath(file) for file in files]
    # Load files from cmd args
    else:
        files = args.files
        
    # Process the list of files
    if len(files) > 0:
        events_path, telescopes_path = generate_dataset(
            files_path=files, output_folder=args.output, append=args.append
        )
    else:
        raise ValueError(f"folder or files not set correctly. (len(files)={len(files)})")

    # dataset = load_dataset(events_path, telescopes_path)
    
    # print("Dataset")
    # describe_dataset(dataset)
    
    # if split > 0:
    #     train_dataset, val_dataset = split_dataset(dataset, split)

    #     save_dataset(train_dataset, output, "train")
    #     save_dataset(val_dataset, output, "validation")

    #     print("\ntrain_dataset:")
    #     describe_dataset(train_dataset)
        
    #     print("\nval_dataset:")
    #     describe_dataset(val_dataset)

    # if split == 0:

    #     save_dataset(dataset, output, "test")
   
    #     print("\ntest_dataset:")
    #     describe_dataset(dataset)

