import json
import os
from uuid import uuid4
from langchain_core.documents import Document
import typer
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from gotrue.types import AuthResponse
from gotrue import User

from app.utils import QACLILog

from app.database import supabase_instance


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

from app.vectorstores.custom_supabase_vector_store import QACLISupabaseVectorStore

embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-V2")


@dataclass
class QACliLibrary:
    cli_user: "QACliUser"

    def __post_init__(self):
        self.document_file_paths: List[Path] = []
        doc_dir = f"./{self.cli_user.user_knowlesge_base_path}/documents"
        for doc_file_str in os.listdir(doc_dir):
            pdf_file_path = Path(f"./{doc_dir}/{doc_file_str}")

            if "pdf" not in pdf_file_path.suffix.lower():
                QACLILog.error(
                    f"Only PDF files are supported for now. Kindly remove ./{pdf_file_path}"
                )
                raise typer.Abort()
            self.document_file_paths.append(pdf_file_path)

    def list_documents(self):
        if not self.document_file_paths:
            QACLILog.error(
                f"{self.cli_user.first_name} does not have any documents in their knowledge base."
            )
            raise typer.Abort()

        return self.document_file_paths

    def upsert_document_vectors(self):
        for pdf_file_path in self.list_documents():
            loader = PyPDFLoader(f"{pdf_file_path.absolute()}")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, add_start_index=True
            )
            splits = text_splitter.split_documents(docs)

            splits = [
                Document(
                    page_content=doc_split.page_content,
                    metadata={
                        "email": self.cli_user.email,
                        **doc_split.metadata,
                    },
                )
                for doc_split in splits
            ]

            vector_store = QACLISupabaseVectorStore(
                client=supabase_instance,
                table_name="documents",
                query_name="match_documents",
                chunk_size=500,
                embedding=embeddings,
            )

            # Deletes existing vectors for this particular document.
            vector_store.delete_matched_metadata_values(
                meta_data_key_and_values=
                # metadata keys and values
                {
                    "source": splits[0].metadata.get("source", ""),
                    "email": self.cli_user.email,
                }
            )

            embeds = embeddings.embed_documents([doc.page_content for doc in splits])
            embeds_ids = [str(uuid4()) for _ in splits]
            vector_store.add_vectors(embeds, splits, embeds_ids)

            query = "What does the document said about Communications?"

            filter_params = {
                "email": self.cli_user.email,
                # "source_file":
            }
            matched_docs = vector_store.similarity_search_with_relevance_scores(
                query,
                k=5,
                filter=filter_params,
            )
            # for match_doc in matched_docs:
            #     print("\n", match_doc[0].page_content)

            # print("*" * 70, "\n\n")


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
