from enum import Enum
from typing import Any, List, Optional
from collections import OrderedDict
import numpy as np
import tables


class Task(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


class Event:
    """An output sample, it contains the event information including the ground
    truth for regression and classification.
    This structure is the common output (ground truth) for each model (and
    ensembler model) on gerumo.
    The mapper functions should transform the "raw" information into an
    Event during training.
    """
    def __init__(self, event_unique_id: int, energy: float,
                 **kwargs: Any) -> None:
        """
        Args:
            event_unique_id : Dataset identifier.
            kwargs: fields to add to this `Event`.
        """
        self._event_unique_id = event_unique_id
        self._energy = energy
        self._fields: OrderedDict[str, Any] = OrderedDict()
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def event_unique_id(self) -> int:
        """
        Returns:
            int
        """
        return self._event_unique_id

    @property
    def energy(self) -> float:
        """
        Returns:
            float
        """
        return self._energy

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name)) # noqa E501
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> OrderedDict[str, Any]:
        """
        Returns:
            Orderdict: a dict which maps names (str) to data of the fields
        Modifying the returned dict will modify this instance.
        """
        return self._fields

    def __repr__(self) -> str:
        fields = ", ".join([f"{k}: {v}" for k, v in self._fields.items()])
        return f"Event(id={self._event_unique_id}, energy={self._energy}, fields={fields})" # noqa

    def to_tensor(self, fields=None) -> np.ndarray:
        fields = fields or list(self._fields.keys())
        return np.array([self._fields[field] for field in fields])

    @classmethod
    def list_to_tensor(cls, events: List['Event'],
                       fields: Optional[List[str]] = None) -> np.ndarray:
        return np.stack([event.to_tensor(fields) for event in events])


class Observations:
    """An input sample, it contains a list the features (or image)
    corresponding to one of all the observations of an event.
    For Mono reconstruction it contains a list of length 1.
    For Stereo reconstruction, inputs are grouped by the telescope type.
    """
    pass


class Telescope:
    geometries_paths = {
      'LSTCam': '/configuration/instrument/telescope/camera/geometry_LSTCam',
      'FlashCam': '/configuration/instrument/telescope/camera/geometry_FlashCam',  # noqa
      'CHEC': '/configuration/instrument/telescope/camera/geometry_CHEC',
      'NectarCam': '/configuration/instrument/telescope/camera/geometry_NectarCam' # noqa
    }

    def __init__(self, name, type, camera_type):
        self.name = name
        self.type = type
        self.camera_type = camera_type
        self._pixels_positions = None
        self._aligned_pixels_positions = None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "_".join(self.description)

    @property
    def description(self):
        return (self.name, self.type, self.camera_type)

    def set_geometry(self, hdf5_filepath):
        geometry_path = Telescope.geometries_paths[self.camera_type]
        hdf5_file = tables.open_file(hdf5_filepath, "r")
        geometry = hdf5_file.root[geometry_path]
        self._pixels_positions = np.array([(row["pix_x"], row["pix_y"]) for row in geometry]).T  # noqa
        hdf5_file.close()
        self._aligned_pixels_positions = self.__compute_alignment(self._pixels_positions)  # noqa

    def get_pixels_positions(self, hdf5_filepath=None):
        if self._pixels_positions is None:
            assert hdf5_filepath is not None
            self.set_geometry(hdf5_filepath)
        return self._pixels_positions

    def get_aligned_pixels_positions(self, hdf5_filepath=None):
        if self._aligned_pixels_positions is None:
            assert hdf5_filepath is not None
            self.set_geometry(hdf5_filepath)
        return self._aligned_pixels_positions

    def __compute_alignment(self, pixels_positions: np.ndarray) -> np.ndarray:
        from .camera_align import compute_alignment
        return compute_alignment(self.camera_type, pixels_positions)
