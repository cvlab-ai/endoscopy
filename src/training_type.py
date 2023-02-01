from enum import Enum


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class TrainingType(ExtendedEnum):
    BINARY_SEG = "binary-seg"
    MULTILABEL_SEG = "multilabel-seg"
    MULTILABEL_CLASSIFICATION = "multilabel-classification"

    def __str__(self):
        return self.value.lower()