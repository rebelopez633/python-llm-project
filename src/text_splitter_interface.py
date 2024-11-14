from abc import ABC, abstractmethod
class TextSplitterInterface:
    @abstractmethod
    def split_text(text: str) -> list[str]:
        pass
