from abc import ABC, abstractmethod
from typing import List, Dict

class AbstractClassMapper(ABC):
    @abstractmethod
    def map(self, name: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def is_positive(self, name: str) -> bool:
        raise NotImplementedError

class DictClassMapper(AbstractClassMapper):

    def __init__(self, mappings: Dict) -> None:
        super().__init__()
        self.class_mappings = {}
        self.positive_classes = set()

        for output_class, configuration in mappings.items():
            if 'positive' in configuration and configuration['positive'] == True:
                self.positive_classes.add(output_class)
            for input_class in configuration['classes']:
                if input_class in self.class_mappings:
                    self.class_mappings[input_class].append(output_class)
                else:
                    self.class_mappings[input_class] = [output_class]
            

    def map(self, name: str) -> List[str]:
        return self.class_mappings[name] if name in self.class_mappings else []

    def is_positive(self, name: str) -> bool:
        return name in self.positive_classes


class DummyClassMapper(AbstractClassMapper):
    def map(self, name: str) -> List[str]:
        return [name]
    
    def is_positive(self, name: str) -> bool:
        return True
    
