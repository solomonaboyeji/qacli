# QACLI -> Question and Answering Command Line Interface

## supabase
This project fully depends on supabase for storing regular data and embeddings.

#### service_role key as SUPABASE_KEY
The service_role key is needed to be able to perform admin level tasks such as listing users, deleting users. 
Do not expose this in where users would see this, it is better this CLI is ran on the server end not the frontend.

#### Disable confirmation of email account at the supabase server end.

#### Setup supabase for vector embeddings
https://python.langchain.com/docs/integrations/vectorstores/supabase 
```sql

-- THIS IS FOR HUGGING FACE EMBEDDING MODEL: all-MiniLM-L6-V2
-- You need to change the dimension to suit the model you will be 
--- making use to embed your text
--- 1536 works for OpenAI
--- 384 for all-MiniLM-L12-v2

-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table to store your documents
create table
  documents (
    id uuid primary key,
    content text, -- corresponds to Document.pageContent
    metadata jsonb, -- corresponds to Document.metadata
    embedding vector (384) -- 1536 works for OpenAI embeddings, change if needed
  );

-- Create a function to search for documents
create function match_documents (
  query_embedding vector (384),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding;
end;
$$;

```