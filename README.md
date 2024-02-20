# QACLI -> Question and Answering Command Line Interface

## supabase
This project fully depends on supabase for storing regular data and embeddings.

#### service_role key as SUPABASE_KEY
The service_role key is needed to be able to perform admin level tasks such as listing users, deleting users. 
Do not expose this in where users would see this, it is better this CLI is ran on the server end not the frontend.

#### Disable confirmation of email account at the supabase server end.