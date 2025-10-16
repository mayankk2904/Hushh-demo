db_name = "aicup"
db_host = "localhost"
db_user = "postgres"
db_pass = "Moon@2904"
db_port = 5432


# Queries
get_all_tables = "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"  # change this to get info from datasets table

create_table_component = """CREATE TABLE IF NOT EXISTS components (
    id SERIAL PRIMARY KEY,
    part_number TEXT UNIQUE NOT NULL
);"""

create_table_datasets = """CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    component_id INT REFERENCES components(id) ON DELETE CASCADE,
    version INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(component_id, version)
);"""

create_table_image = """CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    dataset_id INT REFERENCES datasets(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    part_number TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW()
);
"""

insert_table_component = "INSERT INTO components (part_number) VALUES ('{}') ON CONFLICT (part_number) DO NOTHING;"
insert_table_datasets = (
    "INSERT INTO datasets (component_id, version) VALUES ((SELECT id FROM components WHERE part_number = '{}'), '{}') ON CONFLICT (component_id, version) DO NOTHING;"
)
insert_table_images = """INSERT INTO images (dataset_id, image_path, part_number)
VALUES (
    (SELECT id FROM datasets 
     WHERE component_id = (SELECT id FROM components WHERE part_number = '{}') 
     AND version = '{}'),
    '{}',
    '{}'
);"""


create_table_users = """CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'employee',
    created_at TIMESTAMP DEFAULT NOW()
);"""
