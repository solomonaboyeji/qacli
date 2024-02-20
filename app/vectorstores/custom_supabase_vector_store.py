from typing import Dict, List
from langchain_community.vectorstores.supabase import SupabaseVectorStore


class QACLISupabaseVectorStore(SupabaseVectorStore):
    """
    A custom VectoreStore that allows the deletion of rows that match a given keys and values in the metadata
    """

    def delete_matched_metadata_values(
        self, meta_data_key_and_values: Dict[str, str]
    ) -> None:
        """Deletes vectors by their metadata values

        Args:
            meta_data_key_and_values (List[Tuple[str, str]]): A list of keys and values to compare with
        """

        self._client.from_(self.table_name).delete().contains(
            "metadata", meta_data_key_and_values
        ).execute()
