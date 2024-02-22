import json
import os
import typer
import shutil

from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union

from gotrue.types import AuthResponse
from gotrue import User
from app.errors import QACLIForgivableError

from app.utils import DocumentInfo, QACLILog, RunMode
from app.database import supabase_instance
from langchain_core.embeddings import Embeddings


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts.chat import ChatPromptTemplate
from app.vectorstores.custom_supabase_vector_store import QACLISupabaseVectorStore
from langchain_community.llms.ollama import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document


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

    def _create_file_ref(self):
        file_ref_id = str(uuid4())
        supabase_instance.table("file_ref").insert(
            {
                "id": file_ref_id,
                "title": "",
                "description": "",
                "user_id": self.cli_user.id,
            }
        ).execute()
        return file_ref_id

    def _generate_document_title_and_description(
        self, vector_store: QACLISupabaseVectorStore
    ):
        query = """
        
        Using the following context alone, answer the question after and return back a JSON format as your answer. 
        
        {context}
        
        {input}
        
        JSON keys should be title and description. In the description field, 
        always start with 'In this document, I will help you answer questions around ... '
        """

        prompt = ChatPromptTemplate.from_template(query)
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)

        retrieval_chain = create_retrieval_chain(
            vector_store.as_retriever(K=4), document_chain
        )
        output = retrieval_chain.invoke(
            {
                "input": "Suggest a suitable title for this document, and a 50 or fewer words of what you can do with the document content"
            }
        )

        return DocumentInfo.model_validate_json(output["answer"])

    def upsert_document_vectors(self, embeddings: Embeddings):
        self.document_info_llm = "openchat"

        self.llm = Ollama(model=self.document_info_llm)

        for pdf_file_path in self.list_documents():
            file_ref_id = self._create_file_ref()
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
                        "file_ref_id": file_ref_id,
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
                chunk_size=300,
                embedding=embeddings,
            )

            # Deletes existing vectors for this particular document if there are any
            vector_store.delete_matched_metadata_values(
                meta_data_key_and_values={
                    "source": splits[0].metadata.get("source", ""),
                    "email": self.cli_user.email,
                }
            )

            # embed the text of the documents
            embeds = embeddings.embed_documents([doc.page_content for doc in splits])
            embeds_ids = [str(uuid4()) for _ in splits]
            vector_store.add_vectors(embeds, splits, embeds_ids)

            document_info = self._generate_document_title_and_description(vector_store)

            # Update the file_ref with these document info
            supabase_instance.table("file_ref").update(
                {
                    "title": document_info.title,
                    "description": document_info.description,
                }
            ).eq("id", file_ref_id).execute()

            supabase_instance.table("documents").update(
                {"file_ref_id": file_ref_id}
            ).eq("metadata->>file_ref_id", file_ref_id).execute()

            # filter_params = {
            #     "email": self.cli_user.email,
            #     # "source_file":
            # }
            # matched_docs = vector_store.similarity_search_with_relevance_scores(
            #     query,
            #     k=5,
            #     filter=filter_params,
            # )
            # for match_doc in matched_docs:
            #     print("\n", match_doc[0].page_content)


@dataclass
class QACliUser:
    """Class for keeping track of a user in the database"""

    email: str
    first_name: str | None
    last_name: str | None
    password: str = "worldsecret!"
    extra: User | None = None
    user_knowlesge_base_path: Path | None = None
    user_knowlesge_base_documents_path: Path | None = None

    id: Union[str, None] = None

    def __post_init__(self):
        # initialising the knowledge base path. This does not mean the file/folder will be existing.
        if self.user_knowlesge_base_path is None:
            self.user_knowlesge_base_path = Path(f"./app/knowledge_base/{self.email}")
            self.user_knowlesge_base_documents_path = Path(
                f"{self.user_knowlesge_base_path.absolute()}/documents"
            )

    @staticmethod
    def list_all_users(email: str | None = None):
        users = [
            QACliUser(
                email=db_user.email,  # type: ignore
                first_name=db_user.user_metadata.get("first_name"),
                last_name=db_user.user_metadata.get("last_name"),
                id=db_user.id,
            )
            for db_user in supabase_instance.auth.admin.list_users()
        ]

        if email:
            users = [_cli_user for _cli_user in users if _cli_user.email == email]

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

        self.user_knowlesge_base_documents_path = Path(
            f"{self.user_knowlesge_base_path}/documents"
        )

        os.makedirs(self.user_knowlesge_base_documents_path, exist_ok=True)
        Path(f"{self.user_knowlesge_base_path}/analyse.json").write_text(
            json.dumps(
                {
                    "questions": [],
                    "focus": "",
                    "sample_questions": [
                        "What part of this document can help my focus?",
                        "Add more questions, go ahead!",
                    ],
                    "sample_focus": "I struggle with communicating with my friends.",
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

    def has_uploaded_knowledge_base(self, run_mode: RunMode | None = None):
        if not run_mode:
            raise QACLIForgivableError("A run mode needs to be selected")

        documents_uploaded = False
        if len(os.listdir(self.user_knowlesge_base_documents_path)) > 0:
            documents_uploaded = True

        # If the user wants to analyse, well they should have
        # added at least one question for me to analyse, init?
        if run_mode == RunMode.ANALYSE:
            analyser_content = json.loads(
                Path(f"{self.user_knowlesge_base_path}/analyse.json").read_text()
            )
            if "questions" not in analyser_content or "focus" not in analyser_content:
                QACLILog.error(
                    "You might have tampered with the config file so much you omitted the questions or the focus."
                )
                typer.Abort()

        if documents_uploaded:
            return

        QACLILog.error("I solely depend on your knowledge base, it seems it is empty!")
        typer.Abort()
