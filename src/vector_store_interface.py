class VectorStoreInterface:

    def create_vector_store(self, pdf_paths: list[str], urls: list[str]) -> any:
        pass

    def load_vector_store(self) -> any:
        pass