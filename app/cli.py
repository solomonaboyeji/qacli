from datetime import datetime
import json
import os
import pdfkit
import typer
import shutil

from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union
import textract

from gotrue.types import AuthResponse
from gotrue import User
from app.errors import ERROR_CODES, QACLIForgivableError


from app.utils import (
    AnalyserContent,
    DocumentInfo,
    AnalyserAnswer,
    QACLILog,
    RetrieverTechnique,
    RunMode,
    SourceDocument,
    SupportedModel,
    initialise_embeddings_model,
)
from app.database import supabase_instance
from langchain_core.embeddings import Embeddings


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers import LongContextReorder
from langchain.prompts.chat import ChatPromptTemplate
from app.vectorstores.custom_supabase_vector_store import QACLISupabaseVectorStore
from langchain_community.llms.ollama import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.chains import LLMChain, StuffDocumentsChain

# Adding a Contextual Compression with an LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

TOP_K_DOCS = int(os.getenv("TOP_K_DOCS", "4"))
USE_TEXTRACT = os.getenv("USE_TEXTRACT", "False") == "True"


@dataclass
class QACLIAI:
    user: "QACliUser"
    mode: RunMode
    model_name: SupportedModel
    retriever_option: RetrieverTechnique = RetrieverTechnique.DEFAULT_VECTOR_STORE

    def __post_init__(self):
        configuration = (
            f"""Model Name: {self.model_name}\nRetrieval: {self.retriever_option}"""
        )
        QACLILog.info(f"\nUsing the following configuration:\n{configuration}")

    def store_output(self, llm_answers: List[AnalyserAnswer]):
        output_path = Path(f"{self.user.user_knowlesge_base_path}/output")
        if not output_path.exists():
            os.mkdir(output_path.absolute())

        current_date_str = f"{datetime.today()}"
        report_base_path = f"{output_path.absolute()}/{current_date_str}"
        os.makedirs(report_base_path, exist_ok=True)

        output_file_path = Path(f"{report_base_path}/report.json")
        pdf_output_file_path = Path(f"{report_base_path}/report.pdf")

        # save the json format
        output_file_path.write_text(
            json.dumps([llm_answer.dict() for llm_answer in llm_answers])
        )

        html_report = """
        <body>
            <div>
                <bold># QACLI -> Question and Answering Command Line Interface</bold>
                <h1>Analyser Report</h1>
            </div>
        """
        for index, llm_answer in enumerate(llm_answers):

            html_report += f"""
                <main>
                    <div>
                        <h1><bold>Question {index + 1}: {llm_answer.question}</bold> </h1>
                        <p>{llm_answer.answer}</p>
                        <div>
                            <h2>Configuration</h2>
                            <div>
                                <ul>
                                    <li> Model: {llm_answer.chosen_model_name} </li>
                                    <li> Retriever: {llm_answer.retriever} </li>
                                    <li> Number of Relevant Docs: {llm_answer.number_of_relevant_docs} </li>
                                    <li> Base Prompt: {llm_answer.base_prompt} </li>
                                </ul>
                            </div>
                            
                            <div>
                                <h3>Source Documents</h3>
                                <div>
                                    __LIST__SOURCE_DOCUMENTS__
                                </div>
                            </div>
                        </div> 
                        <hr />
                    </div>
                </main>
            </body>
            """

            source_docs_str = ""
            for source_document in llm_answer.source_documents:
                source_docs_str += f"""
                    <div  >
                        <ul>
                            <li> Title: {source_document.title} </li>
                            <li> Description: {source_document.description} </li>
                            <li> Excerpt: {source_document.piece} </li>
                            <li> <small><i>#{source_document.ref_id}</i></small> </li>
                        </ul>
                    </div>
                """

            html_report = html_report.replace(
                "__LIST__SOURCE_DOCUMENTS__", source_docs_str
            )

        pdfkit.from_string(html_report, pdf_output_file_path)

        QACLILog.success(f"PDF and JSON reports saved to {output_file_path.name}")

    def analyse_user_documents(self):
        QACLILog.warning("Analysing...")

        # load the questions from the analyse.json
        if not self.user.user_analyser_path.exists():
            QACLILog.error(
                f"{self.user.email} does not have analyse.json their knowledge base."
            )
            raise typer.Abort()

        analyser_content = AnalyserContent.model_validate_json(
            self.user.user_analyser_path.read_text(encoding="utf-8")
        )

        base_prompt_str = """
            You are a professional assistant that is versatile in so many areas of life.
            
            You go straight to the answer without wasting time when replying.
            
            Your task is to use the context below to answer their question. 
        
            [Context Starts]
            {context}
            [Context Ends]

            [Question Starts]:
            Using the context above: Goal: '#--focus--#'. {question}.
            [Question Ends]
            
            
            
            """

        # Use the following context to answer the following question aftewards.

        # [Context Starts]
        # Focus: #--focus--#
        # {context}
        # [Context Ends]

        # [Question Starts]:
        # {question}.
        # [Question Ends]

        # Your answer should be concise and meangingful to the focus.
        # If your answer does not come from the context, do NOT come up with any other answer!!
        # """

        embeddings = initialise_embeddings_model()
        llm_answers = []
        vector_store = QACLISupabaseVectorStore(
            client=supabase_instance,
            table_name="documents",
            query_name="match_documents",
            chunk_size=300,
            embedding=embeddings,
        )

        document_variable_name = "context"
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        llm = Ollama(model=self.model_name)
        # base_prompt_str = base_prompt_str.replace(
        #     "#--limited_knowledge_error--#",
        #     ERROR_CODES["limited_knowledge_error"]["code"],
        # )
        base_prompt_str = base_prompt_str.replace("#--focus--#", analyser_content.focus)

        prompt = PromptTemplate(
            input_variables=["question", "question"],
            template=base_prompt_str,
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )

        for question in analyser_content.questions:
            print("\n")
            print(question, "\n")

            relevant_docs = []

            def get_relevant_docs(retriever, question: str, top_k: int = 20):

                filter_params = {"email": self.user.email}
                return retriever.get_relevant_documents(
                    question, k=top_k, filter=filter_params
                )

            if self.retriever_option == RetrieverTechnique.LONG_CONTENT_REORDER:
                relevant_docs = get_relevant_docs(
                    retriever=vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": TOP_K_DOCS, "score_threshold": 0.01},
                    ),
                    question=question,
                )
                reordering = LongContextReorder()
                relevant_docs = reordering.transform_documents(relevant_docs)
            elif self.retriever_option == RetrieverTechnique.DEFAULT_VECTOR_STORE:
                # use the default vector store
                pass
            elif (
                self.retriever_option
                == RetrieverTechnique.CONTEXTUAL_COMPRESSOR_RETRIEVER
            ):
                compressor = LLMChainExtractor(llm_chain=llm_chain)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": TOP_K_DOCS, "score_threshold": 0.01},
                    ),
                )
                relevant_docs = get_relevant_docs(
                    retriever=compression_retriever, question=question
                )
            elif (
                self.retriever_option
                == RetrieverTechnique.CONTEXTUAL_LLMCHAIN_FILTER_RETRIEVER
            ):

                _filter = LLMChainFilter(llm_chain=llm_chain)
                embeddings_filter_retriever = ContextualCompressionRetriever(
                    base_compressor=_filter,
                    base_retriever=vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": TOP_K_DOCS, "score_threshold": 0.01},
                    ),
                )
                relevant_docs = get_relevant_docs(
                    retriever=embeddings_filter_retriever, question=question
                )
            elif (
                self.retriever_option
                == RetrieverTechnique.CONTEXTUAL_EMBEDDINGS_FILTER_RETRIEVER
            ):

                _filter = EmbeddingsFilter(
                    embeddings=embeddings, similarity_threshold=0.76
                )
                embeddings_filter_retriever = ContextualCompressionRetriever(
                    base_compressor=_filter,
                    base_retriever=vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": TOP_K_DOCS, "score_threshold": 0.01},
                    ),
                )
                relevant_docs = get_relevant_docs(
                    retriever=embeddings_filter_retriever, question=question
                )
            elif (
                self.retriever_option
                == RetrieverTechnique.EMBEDDINGS_AND_DOCUMENT_COMPRESSORS_RETRIEVER
            ):

                splitter = CharacterTextSplitter(
                    chunk_size=300, chunk_overlap=0, separator=". "
                )
                redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
                relevant_filter = EmbeddingsFilter(
                    embeddings=embeddings, similarity_threshold=0.76
                )
                pipeline_compressor = DocumentCompressorPipeline(
                    transformers=[splitter, redundant_filter, relevant_filter]
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=pipeline_compressor,
                    base_retriever=vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": TOP_K_DOCS, "score_threshold": 0.01},
                    ),
                )
                relevant_docs = get_relevant_docs(
                    retriever=compression_retriever, question=question
                )

            else:
                QACLILog.error(
                    f"Unknown retriever option selected. ({self.retriever_option})"
                )
                raise typer.Abort()

            output = chain.invoke(
                {"question": question, "input_documents": relevant_docs}
            )

            print(output["output_text"])
            print("\n\n")

            # identify the title of the documents where answers are coming from
            source_docs = []
            for input_doc in output["input_documents"]:
                file_ref_id = input_doc.metadata["file_ref_id"]
                start_index = input_doc.metadata["start_index"]
                piece = input_doc.page_content
                file_ref = (
                    supabase_instance.table("file_ref")
                    .select("id, title, description")
                    .eq("id", file_ref_id)
                    .limit(1)
                    .execute()
                )
                if file_ref and file_ref.data:
                    source_doc_title = file_ref.data[0].get("title")
                    source_doc_description = file_ref.data[0].get("description")
                    file_ref_id = file_ref.data[0].get("id")
                    source_docs.append(
                        SourceDocument(
                            title=source_doc_title,
                            description=source_doc_description,
                            ref_id=file_ref_id,
                            piece=piece,
                            start_index=start_index,
                        )
                    )

            llm_answers.extend(
                [
                    AnalyserAnswer(
                        answer=str(output["output_text"]).replace("\n", "<br />"),
                        chosen_model_name=self.model_name,
                        retriever=self.retriever_option.value,
                        number_of_relevant_docs=len(relevant_docs),
                        question=question,
                        base_prompt=base_prompt_str,
                        source_documents=source_docs,
                    )
                ]
            )

        self.store_output(llm_answers)

    def chat_with_user_documents(self):
        QACLILog.warning("Chat initiated...")

    def run(self):
        if self.mode == RunMode.ANALYSE:
            self.analyse_user_documents()
        elif self.mode == RunMode.CHAT:
            self.chat_with_user_documents()
        else:
            QACLILog.error(f"{self.mode} run mode not yet supported.")


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
                f"{self.cli_user.email} does not have any documents in their knowledge base."
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
        Use the following context to answer the following question aftewards. 
        
        [Context Starts]
        {context}
        [Context Ends]

        [Question Starts]:
        {input}.
        [Question Ends]
        
        
        Your answer should be a JSON format with title and description as key. Do not add any backticks or markdowns, just plain JSON string

        Your answer should be concise and meangingful to the context.
        """

        prompt = ChatPromptTemplate.from_template(query)
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)

        retrieval_chain = create_retrieval_chain(
            vector_store.as_retriever(
                # searcg_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.01, "k": TOP_K_DOCS},
            ),
            document_chain,
        )
        output = retrieval_chain.invoke(
            {
                "input": "What is this document all about? Suggest a title and description for this document"
            }
        )

        return DocumentInfo.model_validate_json(output["answer"])

    def upsert_document_vectors(self, embeddings: Embeddings):
        """This function will update the embeddings of this particular user
        in the database. If the user does not have a matched document, the function
        will create new ones. Please note this function is likely running in another thread
        or processor.

        Args:
            embeddings (Embeddings): The embeddings to upsert.
        """
        self.document_info_llm = "mistral:7b"

        self.llm = Ollama(model=self.document_info_llm)

        documents_list = self.list_documents()

        for pdf_file_path in documents_list:

            file_ref_id = self._create_file_ref()

            if USE_TEXTRACT:
                QACLILog.success("Using textract to read the documents")
                text_corpus = str(
                    textract.process(
                        pdf_file_path.absolute(),
                        output_encoding=textract.parsers.DEFAULT_OUTPUT_ENCODING,
                    )
                )
                temp_text_path = Path(f"{pdf_file_path.absolute()}.txt")
                temp_text_path.write_text(
                    text_corpus, encoding=textract.parsers.DEFAULT_OUTPUT_ENCODING
                )
                loader = TextLoader(
                    f"{temp_text_path.absolute()}",
                    textract.parsers.DEFAULT_OUTPUT_ENCODING,
                )
                docs = loader.load()
                os.remove(temp_text_path)
            else:
                QACLILog.success("Using default PDF Loader PyPDFLoader")
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
            response = (
                supabase_instance.table("file_ref")
                .update(
                    {
                        "title": document_info.title,
                        "description": document_info.description,
                    }
                )
                .eq("id", file_ref_id)
                .execute()
            )

            response = (
                supabase_instance.table("documents")
                .update({"file_ref_id": file_ref_id})
                .eq("metadata->>file_ref_id", file_ref_id)
                .execute()
            )


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
            self.user_analyser_path = Path(
                f"{self.user_knowlesge_base_path.absolute()}/analyse.json"
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
            f"./{self.user_analyser_path} shold be populated with questions you want to analyse the documents on."
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

            if not analyser_content.get("questions", None) or not analyser_content.get(
                "focus"
            ):
                QACLILog.error(
                    "The analyser needs populated questions and a specific focus."
                )
                typer.Abort()

        if documents_uploaded:
            return

        QACLILog.error("I solely depend on your knowledge base, it seems it is empty!")
        typer.Abort()
