import os
import typer
from app.cli import QACliUser
from app.utils import QACLILog


app = typer.Typer(help="Question and Answering Command Line Interface -> qaCLI.")

DEFAULT_EMAIL = os.getenv("DEFAULT_EMAIL", "bigrag@yourcompany.com")
DEFAULT_FIRST_NAME = os.getenv("DEFAULT_FIRST_NAME", "John")
DEFAULT_LAST_NAME = os.getenv("DEFAULT_LAST_NAME", "Doe")


@app.command()
def create_user(
    first_name: str = DEFAULT_FIRST_NAME,  # type: ignore
    last_name: str = DEFAULT_LAST_NAME,  # type: ignore
    email: str = DEFAULT_EMAIL,  # type: ignore
):
    """
    Create a new user with EMAIL, FIRST NAME, LAST NAME.
    """
    print(f"Creating user: {email}")
    new_user = QACliUser(email, first_name, last_name)
    new_user.create()
    new_user.setup_knowledge_base_directory()
    print("User Created", new_user.id)


@app.command()
def sign_in(email: str = DEFAULT_EMAIL):
    """Signs the user in"""

    user = QACliUser(email=email, first_name=None, last_name=None)
    user.sign_in()
    QACLILog.success(message=f"User {email} signed in", bold=True)


@app.command()
def show_all_users(email: str = DEFAULT_EMAIL):
    """Lists all the users in the database"""

    from rich.table import Table

    table = Table("#", "First Name", "Last Name", "Email")

    QACLILog.success("\n\nShowing the users in the database.")
    for index, cli_user in enumerate(QACliUser.list_all_users()):
        table.add_row(
            f"{index + 1}", cli_user.first_name, cli_user.last_name, cli_user.email
        )
        table.add_section()

    QACLILog.success(table)


@app.command()
def delete_user(email: str = DEFAULT_EMAIL):
    """Deletes the user and their knowledge base. Please be careful as this is not reversible!"""

    user = QACliUser(email=email, first_name=None, last_name=None)
    user.sign_in()
    QACLILog.warning(f"Deleting {user.first_name}-{user.id}")
    user.delete()
    QACLILog.success(message=f"Bye! {email}", bold=True)


if __name__ == "__main__":
    app()
