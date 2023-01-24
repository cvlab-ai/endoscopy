from abc import ABC, abstractmethod
from typing import List, Dict

class AbstractClassMapper(ABC):
    @abstractmethod
    def map(self, name: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self, name: str) -> bool:
        raise NotImplementedError

class DictClassMapper(AbstractClassMapper):

    def __init__(self, mappings: Dict) -> None:
        super().__init__()
        self.class_mappings = {}
        self.healthy_mappings = {}

        for output_class, configuration in mappings.items():
            self.healthy_mappings[output_class] = configuration.get('healthy', False)
            for input_class in configuration['classes']:
                if input_class in self.class_mappings:
                    self.class_mappings[input_class].append(output_class)
                else:
                    self.class_mappings[input_class] = [output_class]
            

    def map(self, name: str) -> List[str]:
        return self.class_mappings[name] if name in self.class_mappings else []

    def is_healthy(self, name: str) -> bool:
        return self.healthy_mappings.get(name, False)


class DummyClassMapper(AbstractClassMapper):

    def __init__(self):
        self.healthy_classes = ['h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07', 'b02']

    def map(self, name: str) -> List[str]:
        return [name]
    
    def is_healthy(self, name: str) -> bool:
        return name in self.healthy_classes
