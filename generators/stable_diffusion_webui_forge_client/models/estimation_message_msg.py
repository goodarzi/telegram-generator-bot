from enum import Enum


class EstimationMessageMsg(str, Enum):
    ESTIMATION = "estimation"

    def __str__(self) -> str:
        return str(self.value)
