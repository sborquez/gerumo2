"""
Data and dataset input and output
=================================

This module handle data files and generate the datasets used
by the models.
"""

from os import path
from glob import glob
from tqdm import tqdm
import uuid
import csv
import tables
import logging
import pandas as pd
import numpy as np

# TODO
_telescope_fieldnames = []
_event_fieldnames = []

def extract_data(hdf5_filepath):
    """Extract data from one hdf5 file."""
    #TODO
    return None, None

    hdf5_file = tables.open_file(hdf5_filepath, "r")
    source = path.basename(hdf5_filepath)
    folder = path.dirname(hdf5_filepath)

    events_data = []
    telescopes_data = []

    # Array data
    array_data = {}
    # Telescopes Ids
    real_telescopes_id = {}
    ## 'activated_telescopes' are not the real id for each telescopes. These activated_telecopes
    ## are indices related to the event table, but not with  the array information table (telescope info).
    ## In the array info table, all telescope (from different types) are indexed together starting from 1.
    ## But they are in orden, grouped by type (lst, mst and then sst). In the other hand, the Event table
    ## has 3 indices starting from 0, for each telescope type. 
    ## 'real_telescopes_id' translate events indices ('activation_telescope_id') to array ids ('telescope_id').

    for telescope in hdf5_file.root[_array_info_table[version]]:
        telescope_type = telescope[_array_attributes[version]["type"]]
        telescope_type = telescope_type.decode("utf-8") if isinstance(telescope_type, bytes) else telescope_type
        telescope_id = telescope[_array_attributes[version]["telescope_id"]]
        # HERE
        if telescope_type not in array_data:
            array_data[telescope_type] = {}
            real_telescopes_id[telescope_type] = []

        array_data[telescope_type][telescope_id] = {
            "id": telescope_id,
            "x": telescope[_array_attributes[version]["x"]],
            "y": telescope[_array_attributes[version]["y"]],
            "z": telescope[_array_attributes[version]["z"]],
        }
        real_telescopes_id[telescope_type].append(telescope_id)

    # add uuid to avoid duplicated event numbers 
    try:
        for i, event in enumerate(hdf5_file.root[_events_table[version]]):
            # Event data
            event_unique_id = uuid.uuid4().hex[:20]
            event_data = dict(
                event_unique_id=event_unique_id,
                event_id=event[_event_attributes[version]["event_id"]],
                source=source,
                folder=folder,
                core_x=event[_event_attributes[version]["core_x"]],
                core_y=event[_event_attributes[version]["core_y"]],
                h_first_int=event[_event_attributes[version]["h_first_int"]],
                alt=event[_event_attributes[version]["alt"]],
                az=event[_event_attributes[version]["az"]],
                mc_energy=event[_event_attributes[version]["mc_energy"]]
            )
            events_data.append(event_data)

            # Observations data
            ## For each telescope type
            for telescope_type in TELESCOPES:
                telescope_type_alias = TELESCOPES_ALIAS[version][telescope_type]
                telescope_indices = f"{telescope_type_alias}_indices"
                telescopes = event[telescope_indices]
                # number of activated telescopes
                if version == "ML2":
                    telescope_multiplicity = f"{telescope_type_alias}_multiplicity"
                    multiplicity = event[telescope_multiplicity]
                else:
                    multiplicity = np.sum(telescopes != 0)

                if multiplicity == 0:  # No telescope of this type were activated
                    continue

                # Select activated telescopes
                activation_mask = telescopes != 0
                activated_telescopes = np.arange(len(telescopes))[activation_mask]
                observation_indices = telescopes[activation_mask]

                ## For each activated telescope
                for activate_telescope, observation_indice in zip(activated_telescopes, observation_indices):
                    # Telescope Data
                    real_telescope_id = real_telescopes_id[telescope_type_alias][activate_telescope]
                    telescope_data = dict(
                        telescope_id=real_telescope_id,
                        event_unique_id=event_unique_id,
                        type=telescope_type,
                        x=array_data[telescope_type_alias][real_telescope_id]["x"],
                        y=array_data[telescope_type_alias][real_telescope_id]["y"],
                        z=array_data[telescope_type_alias][real_telescope_id]["z"],
                        observation_indice=observation_indice
                    )
                    telescopes_data.append(telescope_data)
    except KeyboardInterrupt:
        logging.info("Extraction stopped.")
    except Exception as err:
        logging.error(err)
        logging.info("Extraction ended by an error.")
    else:
        logging.info("Extraction ended successfully.")
    finally:
        logging.debug(f"Total events: {len(events_data)}")
        logging.debug(f"Total observations: {len(telescopes_data)}")

    return events_data, telescopes_data


def generate_dataset(file_paths, output_folder="./output", append=False):
    """Generate events.csv and telescope.csv files. 

    Files generated contains information about the events and their observations
    and are used to reference the compressed hdf5 files with the data.

    Args:
        file_paths (list(str)) : List of path to hdf5 files, use these files
            to generate the dataset.
        output_folder (str, optional) : Path to folder where dataset files 
            will be saved. Defaults to './output'
        append (bool, optional): Append new events and telescopes to existing 
            files, otherwise create new file. Defaults to False.

    Returns: 
        tuple(str): events.csv and telescope.csv path.
    """
    # hdf5 files
    files = [path.abspath(file) for file in file_paths]

    # Check if list is not empty
    if len(files) == 0:
        raise FileNotFoundError
    logging.debug(f"{len(files)} files found.")

    # csv files
    mode = "a" if append else "w"
    events_filepath = path.join(output_folder, "events.csv")
    telescope_filepath = path.join(output_folder, "telescopes.csv")
    events_info_csv = open(events_filepath, mode=mode)
    telescope_info_csv = open(telescope_filepath, mode=mode)

    # csv writers
    telescope_writer = csv.DictWriter(telescope_info_csv, delimiter=";",
                                      fieldnames=_telescope_fieldnames, lineterminator="\n")
    events_writer = csv.DictWriter(events_info_csv, delimiter=';',
                                   fieldnames=_event_fieldnames, lineterminator="\n")

    if not append:
        events_writer.writeheader()
        telescope_writer.writeheader()

    total_events = 0
    total_observations = 0
    for file in tqdm(files):
        logging.info(f"Extracting: {file}")
        events_data, telescopes_data = extract_data(file, version)
        total_events += len(events_data)
        total_observations += len(telescopes_data)
        try:
            events_writer.writerows(events_data)
            telescope_writer.writerows(telescopes_data)
        except KeyboardInterrupt:
            logging.info("Extraction stopped.")
            break
    else:
        logging.info("Extraction ended successfully!")
    logging.info(f"Total events: {total_events}")
    logging.info(f"Total observations: {total_observations}")

    # close files
    telescope_info_csv.close()
    events_info_csv.close()

    return events_filepath, telescope_filepath


def split_dataset(dataset, validation_ratio=0.1):
    """Split dataset in train and validation sets using events and a given ratio. 
    
    This split enforce the restriction of don't mix hdf5 files between sets in a 
    imbalance way, but ignore the balance between telescopes type.
    
    Args:
        dataset (pd.DataFrame) : Generated dataset.
        validation_ratio (float, optional) : Split proportion. Defaults to 0.1

    Returns:
        (pd.DataFrame, pd.DataFrame) : Splited dataset.
    """

    if not (0 < validation_ratio < 1):
        raise ValueError(f"validation_ratio not in (0,1) range: {validation_ratio}")

    # split by events
    total_events = dataset.event_unique_id.nunique()
    val_events_n = int(total_events * validation_ratio)
    train_events_n = total_events - val_events_n

    # enforce source balance
    dataset = dataset.sort_values("source")

    # split by events
    events = dataset.event_unique_id.unique()
    train_events = events[:train_events_n]
    val_events = events[train_events_n:]

    # new datasets
    train_dataset = dataset[dataset.event_unique_id.isin(train_events)]
    val_dataset = dataset[dataset.event_unique_id.isin(val_events)]

    return train_dataset, val_dataset


def load_dataset(events_path, telescopes_path, replace_folder=None):
    """Load events.csv and telescopes.csv files into dataframes.
    
    Args:
        events_path (str) : Path to events.csv file.
        telescopes_path (str) : Path to telescopes.csv file.
        replace_folder (str, optional) : Path to folder containing hdf5 files.
            Replace the folder column from csv file. Usefull if the csv files
            are shared between different machines. Default None, means no change
            applied. Defaults to None.

    Returns:
        pd.DataFrame : Dataset of observations for reference telescope images.
    """
    # Load data
    events_data = pd.read_csv(events_path, delimiter=";")
    telescopes_data = pd.read_csv(telescopes_path, delimiter=";")

    # Change dataset folder
    if replace_folder is not None:
        events_data.folder = replace_folder

    # Join tables
    dataset = pd.merge(events_data, telescopes_data, on="event_unique_id", validate="1:m")

    return dataset


def save_dataset(dataset, output_folder, prefix=None):
    """Save events and telescopes dataframes in the corresponding csv files.
    
    Args:
        dataset (pd.Dataframe): Dataset of observations for reference telescope images.
        output_folder (str): Path to folder where dataset files will be saved.
        prefix (str, optional): Add a prefix to output files names. Defaults to None.

    Returns:
        tuple(str): events.csv and telescope.csv path.
    """

    event_drop = [field for field in _telescope_fieldnames if field != 'event_unique_id']
    telescope_drop = [field for field in _event_fieldnames if field != 'event_unique_id']

    telescope_data = dataset.drop(columns=telescope_drop)
    event_data = dataset.drop(columns=event_drop)
    event_data = event_data.drop_duplicates()

    event_path = "event.csv" if prefix is None else f"{prefix}_events.csv"
    event_path = path.join(output_folder, event_path)

    telescope_path = "telescopes.csv" if prefix is None else f"{prefix}_telescopes.csv"
    telescope_path = path.join(output_folder, telescope_path)

    event_data.to_csv(event_path, sep=";", index=False)
    telescope_data.to_csv(telescope_path, sep=";", index=False)

    return event_path, telescope_path


def describe_dataset(dataset, save_to=None):
    """Print a description of the dataset

    Args:
        dataset (pd.DataFrame): loaded dataset from csv files
        save_to (str, optional): Path to description txt file. Defaults to None.
    """
    files = dataset.source.nunique()
    events = dataset.event_unique_id.nunique()
    obs = len(dataset)
    by_telescope = dataset.type.value_counts()
    print('files', files)
    print('events', events)
    print('observations', obs)
    print('obsevation by telescopes')
    print(by_telescope)

    if save_to is not None:
        with open(save_to, 'w') as save_file:
            save_file.write(f'files: {files}\n')
            save_file.write(f'events: {events}\n')
            save_file.write(f'observations: {obs}\n')
            save_file.write('obsevation by telescopes:\n')
            save_file.write(by_telescope.to_string())


def aggregate_dataset(dataset, az=True, log10_mc_energy=True, hdf5_file=True):
    """Perform simple aggegation to targe columns.

    Args:
        dataset (pd.DataFrame): [description]
        az (bool, optional): Translate domain from [0, 2\pi] to [-\pi, \pi]. Defaults to True.
        log10_mc_energy (bool, optional): Add new log10_mc_energy column, with the logarithm values of mc_energy. Defaults to True.
        hdf5_file (bool, optional): Replace source folder. Defaults to True.

    Returns:
        pd.DataFrame: Dataset with aggregate information.
    """
    if az:
        dataset["az"] = dataset["az"].apply(lambda rad: np.arctan2(np.sin(rad), np.cos(rad)))
    if log10_mc_energy:
        dataset["log10_mc_energy"] = dataset["mc_energy"].apply(lambda energy: np.log10(energy))
    if hdf5_file:
        dataset["hdf5_filepath"] = dataset[["folder", "source"]].apply(lambda x: path.join(x[0], x[1]), axis=1)
    return dataset