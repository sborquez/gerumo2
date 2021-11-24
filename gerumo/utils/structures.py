class Event:
    """An output sample, it contains the event information including the ground
    truth for regression and classification.
    This structure is the common output (ground truth) for each model (and ensembler model) on
    gerumo.
    The mapper functions should transform the "raw" information into an
    Event during training.
    """
    pass


class Observations:
    """An input sample, it contains a list the features (or image) 
    corresponding to one of all the observations of an event.
    For Mono reconstruction it contains a list of length 1.
    For Stereo reconstruction, inputs are grouped by the telescope type.
    """
    pass


class Telescope:
    def __init__(self, name, type, camera_type):
        self.name = name
        self.type = type
        self.camera_type = camera_type

    def get_geometry(self):
        pass

    def get_aligned_geometry(self):
        pass
