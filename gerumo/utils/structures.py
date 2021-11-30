class Event:
    """A dataset sample, it contains the event information including the ground
    truth for regression and classification. It also contains a list the 
    features (or image) corresponding to all the observations of the event,
    grouped by the telescope type if it is required for Stereo reconstruction,
    for Mono reconstruction it contains a list of length 1.
    
    This structure is the common input for each model (and ensembler model) on 
    gerumo.
    
    The mapper functions should transform the "raw" information into an
    Event during training.
    """
    pass
