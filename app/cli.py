import json
import os
import typer
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Union

from gotrue.types import AuthResponse
from gotrue import User

from app.utils import QACLILog

from app.database import supabase_instance


@dataclass
class QACliUser:
    """Class for keeping track of a user in the database"""

    email: str
    first_name: str | None
    last_name: str | None
    password: str = "worldsecret!"
    extra: User | None = None
    user_knowlesge_base_path: Path | None = None

    id: Union[str, None] = None

    def __post_init__(self):
        # initialising the knowledge base path. This does not mean the file/folder will be existing.
        if self.user_knowlesge_base_path is None:
            self.user_knowlesge_base_path = Path(f"./app/knowledge_base/{self.email}")

    @staticmethod
    def list_all_users():
        users = [
            QACliUser(
                email=db_user.email,  # type: ignore
                first_name=db_user.user_metadata.get("first_name"),
                last_name=db_user.user_metadata.get("last_name"),
                id=db_user.id,
            )
            for db_user in supabase_instance.auth.admin.list_users()
        ]
        return users

    def _setup_user(self, user: User):
        self.id = user.id
        self.extra = user

        self.first_name = user.user_metadata.get("first_name")
        self.last_name = user.user_metadata.get("last_name")

    def sign_in(self):
        output = supabase_instance.auth.sign_in_with_password(
            {"email": self.email, "password": self.password}
        )
        if output.user is not None:
            self._setup_user(output.user)
            return self

        raise Exception("Unable to sign the user in.")

    def delete(self):
        if self.id is not None:
            supabase_instance.auth.admin.delete_user(self.id)
            self._delete_knowledge_base()
            return

        QACLILog.error("Unable to delete the user.")
        raise typer.Abort()

    def _delete_knowledge_base(self):
        """Deletes the entire knowledge base of this user. This is not reversible!"""

        if self.user_knowlesge_base_path is not None:
            shutil.rmtree(f"./{self.user_knowlesge_base_path}")

    def setup_knowledge_base_directory(self):
        """Create a dedicated folder for this user to upload their documents"""

        if not self.user_knowlesge_base_path:
            QACLILog.error("Knowedge Base Path not well configured.")
            raise typer.Abort()

        user_knowlesge_base_documents_path = Path(
            f"{self.user_knowlesge_base_path}/documents"
        )

        os.makedirs(user_knowlesge_base_documents_path, exist_ok=True)
        Path(f"{self.user_knowlesge_base_path}/analyse.json").write_text(
            json.dumps(
                {
                    "questions": [
                        "What should QACli analyse in your documents?",
                        "Add more questions, go ahead!",
                    ],
                    "focus": "Your focus topic goes here",
                }
            )
        )
        QACLILog.info("\nYour knowledge base directory is ready to be populated.")
        QACLILog.info(
            f"./{self.user_knowlesge_base_path}/analyser.json shold be populated with questions you want to analyse the documents on."
        )

    def create(self):
        try:
            output: AuthResponse = supabase_instance.auth.sign_up(
                {
                    "email": self.email,
                    "password": self.password,
                    "options": {
                        "data": {
                            "first_name": self.first_name,
                            "last_name": self.last_name,
                        },
                    },
                }
            )

            if output.user is not None:
                self._setup_user(output.user)
                return self
        except Exception as e:
            QACLILog.error(str(e))
            raise typer.Abort()

        raise Exception("There was an error creating user.")
