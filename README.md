# QACLI -> Question and Answering Command Line Interface

## PDF

- `textract` can be an option to read in PDF as it is proven to be better in some cases than the default PyPDFLoader
- `pdfkit` will be used to output report, install it below:

  ### Step 1

  ```sh
    $ pip install pdfkit

  ```

  ### Step 2

  ```sh
    # Ubuntu or Debian
    $ sudo apt-get install wkhtmltopdf

    $ WKHTML2PDF_VERSION='0.12.6-1'

    $ sudo apt install -y build-essential xorg libssl-dev libxrender-dev wget
    $ wget "https://github.com/wkhtmltopdf/packaging/releases/download/${WKHTML2PDF_VERSION}/wkhtmltox_${WKHTML2PDF_VERSION}.bionic_amd64.deb"
    $ sudo apt install -y ./wkhtmltox_${WKHTML2PDF_VERSION}.bionic_amd64.deb


  ```

  #### macOS

  ```
    $ brew install homebrew/cask/wkhtmltopdf
  ```

  #### Windows

  Follow the guidelines here https://wkhtmltopdf.org/

## Install Ollama

```sh
ollama pull openchat
ollama pull mistral:7b
ollama pull mistral:instruct
```

## supabase

This project fully depends on supabase for storing regular data and embeddings.

```sh
# Get the code
git clone --depth 1 https://github.com/supabase/supabase

# Go to the docker folder
cd supabase/docker

# Copy the fake env vars
cp .env.example .env

# Pull the latest images
docker compose pull

# Start the services (in detached mode)
docker compose up -d

```

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


-- Create a table to store your document info
create table
  file_ref (
    id uuid primary key,
    user_id uuid references auth.users(id) on delete cascade,
    title text, -- corresponds to DocumentInfo.title
    description text -- corresponds to DocumentInfo.description
  );

-- Create a table to store your documents
create table
  documents (
    id uuid primary key,
    file_ref_id uuid references public.file_ref(id) on delete cascade,
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
