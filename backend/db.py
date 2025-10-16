from typing import List
import psycopg2
from backend import db_config as config
import logging
from fastapi import FastAPI, File, UploadFile

# create fast-api endpoint to handle multiple file upload

# app = FastAPI()

logging.basicConfig(level=logging.INFO)


def connect_db():
    cursor = None
    try:
        conn = psycopg2.connect(
            database=config.db_name,
            host=config.db_host,
            user=config.db_user,
            password=config.db_pass,
            port=config.db_port,
        )

        cursor = conn.cursor()

        logging.info("Database login successful.")
    except Exception as e:
        logging.critical(e)

    return cursor, conn


def execute_query(cursor, conn, pq_query):
    try:
        cursor.execute(pq_query)
        conn.commit()
    except Exception as e:
        logging.error(e)


# create tables in the database
def create_tables_if_not_exists(cursor, conn):

    pq_queries = [
        config.create_table_component,
        config.create_table_datasets,
        config.create_table_image,
    ]

    for pq_query in pq_queries:
        start_idx, end_idx = pq_query.find("EXISTS") + 7, pq_query.find("(") - 1
        table_name = pq_query[start_idx:end_idx]
        execute_query(cursor, conn, pq_query)
        logging.info(f"Table {table_name} was created.")


# insert data to table


def insert_component_data(cursor, conn, part_number):
    pq_sql = config.insert_table_component
    pq_sql = pq_sql.format(part_number)
    execute_query(cursor, conn, pq_sql)


def insert_datasets_data(cursor, conn, component_id, dataset_version):
    pq_sql = config.insert_table_datasets
    pq_sql = pq_sql.format(component_id, dataset_version)
    execute_query(cursor, conn, pq_sql)


def insert_images_data(cursor, conn, part_number, dataset_version, img_path):
    pq_sql = config.insert_table_images
    pq_sql = pq_sql.format(part_number, dataset_version, img_path, part_number)
    execute_query(cursor, conn, pq_sql)


if __name__ == "__main__":
    cursor, conn = connect_db()
    create_tables_if_not_exists(cursor, conn)

    if cursor:
        cursor.close()
        conn.close()