"""
Data and dataset input and output
=================================

This module handle data files and generate the datasets used
by the models.
"""

from os import (makedirs, path)
from shutil import rmtree
from glob import glob
from posix import listdir
from tqdm import tqdm
import uuid
import tables
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# TODO
_telescope_fieldnames = []
_event_fieldnames = []

def extract_data(hdf5_filepath):
    """Extract data from one hdf5 file."""
    #TODO
    mockup1 = pd.DataFrame([{"a": np.random.randint(10), "b": np.random.randint(10)}])
    mockup2 = pd.DataFrame([{"a": np.random.randint(10), "b": np.random.randint(10)}])
    return mockup1, mockup2

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

    return pd.DataFrame(events_data), pd.DataFrame(telescopes_data)


def append_to_parquet_table(dataframe, filepath=None, writer=None):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.
    Args
        dataframe (pd.DataFrame): df to be written in parquet format.
        filepath (str): target file location for parquet file.
        writer (ParquetWriter): object to write pyarrow tables in parquet format.
        ParquetWriter object. This can be passed in the subsequenct method calls to append DataFrame
        in the pyarrow Table
    """
    table = pa.Table.from_pandas(dataframe)
    if writer is None:
        writer = pq.ParquetWriter(filepath, table.schema)
    writer.write_table(table=table)
    return writer


def generate_dataset(file_paths, output_folder, append=False):
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
    logging.debug(f"{len(files)} files found.")
    if len(files) == 0:
        raise FileNotFoundError
    
    # Dataset folders
    events_folder = path.join(output_folder, "events")
    telescopes_folder = path.join(output_folder, "telescopes")
    
    # If overwrite, remove existing files
    if not append and path.exists(output_folder):
        events_files = len(listdir(events_folder)) if path.exists(events_folder) else 0
        telescopes_files = len(listdir(telescopes_folder)) if path.exists(telescopes_folder) else 0 
        logging.warning(f"Removing existing datasets: {events_files + telescopes_files} files")
        rmtree(events_folder)
        rmtree(telescopes_folder)
    
    # If output folder doesnt exists
    if not path.exists(output_folder):
        logging.info("creating folder {telescopes_folder}")
        logging.info("creating folder {events_folder}")
    makedirs(path.join(output_folder, "telescopes"), exist_ok=True)
    makedirs(path.join(output_folder, "events"), exist_ok=True)
    
    # Appending index
    file_counter = len(listdir(events_folder))
    events_filepath = path.join(events_folder, f"events{file_counter}.parquet")
    telescopes_filepath = path.join(telescopes_folder, f"telescopes{file_counter}.parquet")
    
    # Iterate over h5 files 
    total_events = 0
    total_observations = 0
    events_writer, telescope_writer = None, None
    for file in tqdm(files):
        logging.info(f"Extracting: {file}")
        try:
            events_data, telescopes_data = extract_data(file)
            total_events += len(events_data)
            total_observations += len(telescopes_data)
            events_writer = append_to_parquet_table(events_data, filepath=events_filepath, writer=events_writer)
            telescope_writer = append_to_parquet_table(telescopes_data, filepath=telescopes_filepath, writer=telescope_writer)
        except KeyboardInterrupt:
            logging.warning("Extraction stopped.")
            break
    else:
        logging.info("Extraction ended successfully!")
    logging.info(f"Total events: {total_events}")
    logging.info(f"Total observations: {total_observations}")

    # close writers
    if (events_writer is not None) or (telescope_writer is not None):
        events_writer.close()
        telescope_writer.close()
    logging.info(f"Events file: {events_filepath}")
    logging.info(f"Telescopes file: {telescopes_filepath}")
    return events_folder, telescopes_folder


def split_dataset(dataset, validation_ratio=0.1, balanced_files=True):
    """Split dataset in train and validation sets using events and a given ratio. 
    
    This split enforce the restriction of don't mix hdf5 files between sets in a 
    imbalance way, but ignore the balance between telescopes type.
    
    Args:
        dataset (pd.DataFrame) : Generated dataset.
        validation_ratio (float, optional) : Split proportion. Defaults to 0.1
        balanced_files (bool, optional) : Keep same sources files on splits.
    Returns:
        (pd.DataFrame, pd.DataFrame) : Splited dataset.
    """

    if not (0 < validation_ratio < 1):
        raise ValueError(f"validation_ratio not in (0,1) range: {validation_ratio}")

    # split by events
    total_events = dataset.event_unique_id.nunique()
    val_events_n = int(total_events * validation_ratio)
    train_events_n = total_events - val_events_n

    # source balance
    if balanced_files:
        dataset = dataset.sort_values("source")
    else:
        dataset = dataset.sample(frac=1) # shuffle

    # split by events
    events = dataset.event_unique_id.unique()
    train_events = events[:train_events_n]
    val_events = events[train_events_n:]

    # new datasets
    train_dataset = dataset[dataset.event_unique_id.isin(train_events)]
    val_dataset = dataset[dataset.event_unique_id.isin(val_events)]

    return train_dataset, val_dataset


def load_dataset(events_path, telescopes_path, replace_folder=None, merge=True):
    """Load events.csv and telescopes.csv files into dataframes.
    
    Args:
        events_path (str) : Path to events parquet file or folder.
        telescopes_path (str) : Path to telescopes parquet file or folder.
        replace_folder (str, optional) : Path to folder containing hdf5 files.
            Replace the folder column from csv file. Usefull if the csv files
            are shared between different machines. Default None, means no change
            applied. Defaults to None.
        merge (bool, optional) : Return merged dataset.

    Returns:
        pd.DataFrame : Dataset of observations for reference telescope images.
    """
    # Load data
    events_data = pd.read_parquet(events_path)
    telescopes_data = pd.read_parquet(telescopes_path)

    # Change dataset folder
    if replace_folder is not None:
        events_data.folder = replace_folder

    if merge:
        # Join tables
        dataset = pd.merge(events_data, telescopes_data, on="event_unique_id", validate="1:m")
        return dataset
    else:
        return events_data, telescopes_data


def save_dataset(dataset, output_folder, prefix=None):
    """Save events and telescopes dataframes in the corresponding csv files.
    
    Args:
        dataset (pd.Dataframe): Dataset of observations for reference telescope images.
        output_folder (str): Path to folder where dataset files will be saved.
        prefix (str, optional): Add a prefix to output files names. Defaults to None.

    Returns:
        tuple(str): events.csv and telescope.csv path.
    """
    # Unmerge dataset
    event_drop = [field for field in _telescope_fieldnames if field != 'event_unique_id']
    telescope_drop = [field for field in _event_fieldnames if field != 'event_unique_id']
    telescope_data = dataset.drop(columns=telescope_drop)
    event_data = dataset.drop(columns=event_drop)
    event_data = event_data.drop_duplicates()
    # Save events data
    event_path = "event.parquet" if prefix is None else f"{prefix}_events.parquet"
    event_path = path.join(output_folder, event_path)
    pq.write_table(pa.Table.from_pandas(event_data), event_path)
    # Save telescopes data
    telescope_path = "telescopes.parquet" if prefix is None else f"{prefix}_telescopes.parquet"
    telescope_path = path.join(output_folder, telescope_path)
    pq.write_table(pa.Table.from_pandas(telescope_data), telescope_path)
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