from abc import ABC, abstractmethod

class AbstractClassMapper(ABC):
    @abstractmethod
    def map(self, name: str) -> str:
        raise NotImplementedError

class DictClassMapper(AbstractClassMapper):

    def __init__(self, mappings) -> None:
        super().__init__()
        self.mappings = mappings

    def map(self, name: str) -> str:
        return self.mappings[name] if name in self.mappings else None

class DummyClassMapper(AbstractClassMapper):
    def map(self, name: str) -> str:
        return name
