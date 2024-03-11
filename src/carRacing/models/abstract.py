__all__ = ["CarRacingModel"]

from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np


class CarRacingModel(ABC):

    @abstractmethod
    def __init__(self, env) -> None:
        ...

    @abstractmethod
    def predict(self, observation: np.ndarray, **kwargs) -> Union[List, int]:
        ...

    @abstractmethod
    def train(self, **kwargs) -> None:
        ...

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
       ...

    @abstractmethod
    def save(self, model_path: str, **kwargs) -> None:
        ...
