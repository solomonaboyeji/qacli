import os
import typer
from app.cli import QACliUser, QACliLibrary
from app.utils import QACLILog, RunMode, show_break_line

from rich.table import Table


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

    table = Table("#", "First Name", "Last Name", "Email")

    QACLILog.success("\n\nShowing the users in the database.")
    for index, cli_user in enumerate(QACliUser.list_all_users()):
        table.add_row(
            f"{index + 1}", cli_user.first_name, cli_user.last_name, cli_user.email
        )
        table.add_section()

    QACLILog.success(table)


@app.command()
def update_library(email: str = typer.Option(default=None)):
    """Reads all the documents in all users knowledge base and creates embeddings of each of them.
    The function deletes off old data from the database and generate new ones.

    Args:
        email (str, optional): If an email is provided, the script only updates the library for
        that user knowledge base alone . Defaults to None indicating all users.
    """

    all_users = QACliUser.list_all_users(email=email)

    if not all_users:
        QACLILog.error("There are no users in the database.")
        raise typer.Abort()

    QACLILog.warning(f"Updating library for {len(all_users)} users.")

    # TODO: This should likely happen in each user's thread
    for cli_user in all_users:
        qacli_lib = QACliLibrary(cli_user=cli_user)
        qacli_lib.upsert_document_vectors()


@app.command()
def run(
    email: str = typer.Option(
        default=None,
    ),
    mode: RunMode = typer.Option(default=None),
):
    """Starts the application in a mode where the user will choose.

    Args:
        email (str, optional): The user account to run the chosen mode in. Defaults to typer.Option(default=None).
        mode (str, RunMode): A mode to run the application in. Defaults to typer.Option(default=None).
    """

    show_break_line()
    print(
        """Welcome to QACLI!. Please follow the prompts! You gon' have a nice time using QACLI!
    """
    )

    all_users = QACliUser.list_all_users(email=email)

    if not all_users:
        if email is None:
            QACLILog.error("There are no users in the system.")
        else:
            QACLILog.error(f"There are no user with the email {email}.")
        raise typer.Abort()

    if not email:
        while True:
            print("\nSelect the user using the application: \n")
            for index, user in enumerate(all_users):
                print(f"\n{index + 1} - {user.email} ")

            try:
                index_selected = int(input())
                selected_user = all_users[index_selected - 1]
                QACLILog.success(f"\n\nGood! {selected_user.email} active!")
                break
            except ValueError:
                QACLILog.error("Come on!, select an integer value! Try again.\n")
            except IndexError:
                QACLILog.error(
                    f"You should select a number between 1 and {len(all_users)}. Try again.\n"
                )

        show_break_line()

        # Select a mode
        selected_mode = None
        if not mode:
            run_mode_options = [RunMode.ANALYSE, RunMode.CHAT]
            mode_prompt = (
                f"\n[1] {RunMode.ANALYSE.value} \n[2] {RunMode.CHAT.value}: \nChoose: "
            )
            selected_mode = input(mode_prompt)
            while selected_mode not in ["1", "2"]:
                print("What mode do you want to run the application in?: ")
                selected_mode = input(mode_prompt)

            selected_mode = run_mode_options[int(selected_mode) - 1]
            QACLILog.success(f"\nMode: {selected_mode} selected!")

        show_break_line()

        selected_user.has_uploaded_knowledge_base(run_mode=selected_mode)


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
