from typing import Dict
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from supabase.client import Client

from app.utils import QACLILog


class QACLISupabaseVectorStore(SupabaseVectorStore):
    """
    A custom VectoreStore that allows the deletion of rows that match a given keys and values in the metadata
    """

    def add_document_metadata(
        self,
        metadata_to_match: dict,
        existing_metadata: dict,
        new_metadata_to_add: dict,
    ):
        self._client.from_(self.table_name).update(
            {"metadata": {**existing_metadata, **new_metadata_to_add}}
        ).contains("metadata", metadata_to_match).execute()

    def delete_matched_metadata_values(
        self, meta_data_key_and_values: Dict[str, str]
    ) -> None:
        """Deletes vectors by their metadata values

        Args:
            meta_data_key_and_values (List[Tuple[str, str]]): A list of keys and values to compare with
        """

        client: Client = self._client

        # get the first document that matches this and pick the file_ref to delete the file ref immediately
        output = (
            client.table("documents")
            .select("file_ref_id")
            .contains("metadata", meta_data_key_and_values)
            .limit(1)
            .execute()
        ).data

        if output:
            file_ref_id = output[0]["file_ref_id"]

            if file_ref_id:
                QACLILog.info("Deleting existing database file references")
                client.table("file_ref").delete().eq("id", file_ref_id).execute()

        # this should have been cascaded, but let us give it a try. Trust?
        QACLILog.info("Deleting existing database documents that matches")
        output = (
            client.from_(self.table_name)
            .delete()
            .contains("metadata", meta_data_key_and_values)
            .execute()
        )
