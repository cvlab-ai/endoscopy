import os
import shutil
from enum import Enum
from abc import ABC, abstractmethod


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class AbstractCopyStrategy(ABC):
    @abstractmethod
    def copy(self, src: str, dest: str) -> None:
        raise NotImplementedError


class DuplicateCopyStrategy(AbstractCopyStrategy):
    def copy(self, src: str, dest: str) -> None:
        shutil.copy(src, dest)


class SymlinkCopyStrategy(AbstractCopyStrategy):
    def copy(self, src: str, dest: str) -> None:
        os.symlink(src, dest)


class CopyStrategy(ExtendedEnum):
    DUPLICATE = "duplicate"
    SYMLINK = "symlink"

    def __str__(self):
        return self.value.lower()

    def create(self) -> AbstractCopyStrategy:
        return DuplicateCopyStrategy() if self == CopyStrategy.DUPLICATE else SymlinkCopyStrategy()
