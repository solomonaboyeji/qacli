import enum
from typing import Any
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


class RunMode(enum.Enum):
    ANALYSE: str = "ANALYSE"  # type: ignore
    CHAT: str = "CHAT"  # type: ignore


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
