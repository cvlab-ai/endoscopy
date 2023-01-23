from enum import Enum


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class TrainingType(ExtendedEnum):
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"

    def __str__(self):
        return self.value.lower()