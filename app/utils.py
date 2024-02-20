from typing import Any
from rich import print, table
from rich.console import Console


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
