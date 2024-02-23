import enum
from typing import Any, List
from pydantic import BaseModel
from rich import print, table
from rich.console import Console

from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

embeddings_model_name = "all-MiniLM-L6-V2"


def initialise_embeddings_model():
    return HuggingFaceBgeEmbeddings(model_name=embeddings_model_name)


def show_break_line(how_many: int = 100):
    """Print some random lines on the screen!"""
    print("\n", "*" * how_many)


class SupportedModel(enum.StrEnum):
    LLAMA2_7B: str = "llama2:7b"  # type: ignore
    LLAMA2_13B: str = "llama2:13b"  # type: ignore
    OPEN_CHAT: str = "openchat"  # type: ignore
    MISTRAL_INSTRUCT: str = "mistral:instruct"  # type: ignore
    MISTRAL_7B: str = "mistral:7b"  # type: ignore
    LLAVA_v1_6: str = "llava:v1.6"  # type: ignore


class RetrieverTechnique(enum.Enum):
    # Best One
    LONG_CONTENT_REORDER: str = "LONG_CONTENT_REORDER"  # type: ignore
    DEFAULT_VECTOR_STORE: str = "DEFAULT_VECTOR_STORE"  # type: ignore
    CONTEXTUAL_COMPRESSOR_RETRIEVER: str = "CONTEXTUAL_COMPRESSOR_RETRIEVER"  # type: ignore
    # Best One
    CONTEXTUAL_LLMCHAIN_FILTER_RETRIEVER: str = "CONTEXTUAL_LLMCHAIN_FILTER_RETRIEVER"  # type: ignore
    CONTEXTUAL_EMBEDDINGS_FILTER_RETRIEVER: str = "CONTEXTUAL_EMBEDDINGS_FILTER_RETRIEVER"  # type: ignore
    EMBEDDINGS_AND_DOCUMENT_COMPRESSORS_RETRIEVER: str = "EMBEDDINGS_AND_DOCUMENT_COMPRESSORS_RETRIEVER"  # type: ignore


class RunMode(enum.Enum):
    ANALYSE: str = "ANALYSE"  # type: ignore
    CHAT: str = "CHAT"  # type: ignore


class SourceDocument(BaseModel):
    title: str
    description: str
    piece: str
    ref_id: str
    start_index: int


class AnalyserAnswer(BaseModel):
    question: str
    answer: str
    retriever: str
    chosen_model_name: str
    number_of_relevant_docs: int
    base_prompt: str
    source_documents: List[SourceDocument]


class AnalyserContent(BaseModel):
    focus: str
    questions: List[str]


class DocumentInfo(BaseModel):
    title: str
    description: str


class QACLILog:

    @staticmethod
    def success(message: Any, bold: bool = False):
        if isinstance(message, table.Table):
            console = Console()
            console.print(message)
            return

        bolded = "bold " if bold else ""
        print(f"[{bolded}green]{message}[/{bolded}green]")

    @staticmethod
    def error(message: str, bold: bool = False):
        bolded = "bold " if bold else ""
        print(f"[{bolded}red]{message}[/{bolded}red]")

    @staticmethod
    def info(message: str, bold: bool = False):
        bolded = "bold " if bold else ""
        print(f"[{bolded}blue]{message}[/{bolded}blue]")

    @staticmethod
    def warning(message: str, bold: bool = False):
        bolded = "bold " if bold else ""
        print(f"[{bolded}yellow]{message}[/{bolded}yellow]")
