from abc import ABC, abstractmethod

import amanda


class Tool(ABC):
    @abstractmethod
    def instrument(self, graph: amanda.Graph) -> amanda.Graph:
        ...

    def finish(self):
        return
