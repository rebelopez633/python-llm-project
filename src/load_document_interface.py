from abc import ABC, abstractmethod

class LoadDocumentInterface:
    @abstractmethod
    def load_pdf(paths: list[str]) -> list[any]:
        pass

    @abstractmethod
    def load_web(urls: list[str]) -> list[any]:
        pass