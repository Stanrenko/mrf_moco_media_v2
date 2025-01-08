""" model class """
import pathlib
import numpy as np
import pickle
import datetime
from abc import ABC, abstractmethod


class ModelError(ValueError):
    """Wrong model"""


class ModelValueError(ValueError):
    """Required values error"""


def format_values(keys, values):
    """format keys and values"""

    # keys are tuple
    keys = [astuple(key) for key in keys]

    # format values
    if len(keys) != len(values):
        raise ValueError("Mismatch in the number of keys and values")

    if not isinstance(values, np.ndarray):
        values = np.asarray(values)
    if values.ndim == 1:
        # 1-dimensional values
        values = values[..., np.newaxis, np.newaxis]
    elif values.ndim == 2:
        # 1-component values
        values = values[..., np.newaxis]
    elif values.ndim > 3:
        # ndim != 3
        raise ValueError(f"Invalid number of value dimensions: {values.ndim}")

    # return
    return keys, values


def astuple(item):
    """return item as tuple"""
    if isinstance(item, (list, tuple, np.ndarray)):
        return tuple(item)
    return (item,)


def select_model(filename, models):
    """try loading models in sequence"""
    filename = pathlib.Path(filename)
    if not filename.is_file():
        raise FileNotFound(filename)

    payload = None
    for modelclass in models:
        if payload is None:
            try:
                payload = modelclass.__LOADER__(filename)
            except:
                continue

        model = modelclass()
        try:
            model.load(payload)
            return model
        except ModelError:
            pass
    else:
        raise ModelError(f"No matching model for {filename}")


def pickle_save(filename, payload, overwrite=False):
    """pickle payload"""
    filename = pathlib.Path(filename)
    if not overwrite and filename.is_file():
        raise FileExistsError(f"File already exists: {filename}")

    with open(filename, "wb") as fp:
        pickle.dump(payload, fp)


def pickle_load(filename):
    """unpickle payload"""
    with open(filename, "rb") as fp:
        return pickle.load(fp)



"""
    solver = RegressionSolver()

    # build model
    solver.setup(...)

    # save/load model
    solver.save(filename)
    solver.load(filename)

"""


class RegressionModelABC(ABC):
    """Abstract namespace class for regression model"""

    @abstractmethod
    def save(self, filename):
        """save model to file"""

    @abstractmethod
    def load(self, filename):
        """load model from file"""


class SearchResult:
    """Simple namespace to store the results"""

    def __init__(self):
        self.parameters = None
        self.scales = None
        self.info = None
        self.timestamp = datetime.datetime.now()


class RegressionSolver(ABC):
    """Abstract class for defining regression solvers"""

    def __init__(self, options=None):
        self.model = None
        self.options = {} if not options else options
        self.info = {}

    # i/o
    @abstractmethod
    def load(self, filename, **kwargs):
        """load model from file"""

    def save(self, filename, **kwargs):
        """save model to file"""
        self.model.save(filename, **kwargs)

    # core
    @property
    @abstractmethod
    def size_parameters(self):
        """return size of parameter vector"""

    @property
    @abstractmethod
    def size_observations(self):
        """return size of observation vector"""

    @property
    @abstractmethod
    def num_components(self):
        """number of components"""

    @abstractmethod
    def setup(self, parameters, components, **options) -> RegressionModelABC:
        """build model from parameters and components"""

    @abstractmethod
    def search(self, observations, **options) -> SearchResult:
        """regress parameters and scales from observations"""

    @abstractmethod
    def predict(self, res: SearchResult):
        """generate observations from search result"""


class RegressionModel(RegressionModelABC):
    """Basic pickle container"""

    metadata = None
    __REQUIRED__ = []
    __LOADER__ = staticmethod(pickle_load)
    __SAVER__ = staticmethod(pickle_save)

    @property
    def name(self):
        return type(self).__name__

    @property
    def required(self):
        return type(self).__REQUIRED__

    def save(self, filename, **kwargs):
        """save model to file"""
        # payload: all public data
        payload = vars(self)

        # check required
        if not set(self.required) <= set(payload):
            missing = set(self.required) - set(payload)
            raise ModelValueError(f"Missing required data: {missing}")

        # add metadata
        payload["__NAME__"] = self.name
        payload["__REQUIRED__"] = self.required

        self.__SAVER__(filename, payload, **kwargs)

    def load(self, obj, force=False, **kwargs):
        """load dictionary model from file"""

        if isinstance(obj, dict):
            payload = obj
        else:
            payload = self.__LOADER__(obj, **kwargs)

        # check name
        if force != True:
            name = payload.get("__NAME__")
            if name != self.name:
                raise ModelError(f"Model type mismatch: {name} != {self.name}")

            # check required
            if not set(self.required) <= set(payload):
                missing = set(self.required) - set(payload)
                raise ModelValueError(f"Missing required data: {missing}")

        # store values
        for var in payload:
            if var.startswith("_"):
                continue
            setattr(self, var, payload[var])


class Dictionary(RegressionModel):
    """key/value container"""

    __NAME__ = "Dictionary"
    __REQUIRED__ = ["keys", "values"]

    def __init__(self, keys=None, values=None):
        super().__init__()
        if keys is not None:
            keys, values = format_values(keys, values)
            self.keys = keys
            self.values = values

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return key in self.keys

    def __iter__(self):
        return iter(self.keys)

    _dict = None

    def __getitem__(self, key):
        """map key to value"""
        if self._dict is None:
            self._dict = dict(zip(self.keys, self.values))
        return self._dict[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
