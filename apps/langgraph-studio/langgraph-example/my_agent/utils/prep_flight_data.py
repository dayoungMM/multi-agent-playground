import os
import sqlite3

import pandas as pd
import requests

DB_FILE_PATH = "./travel2.sqlite"


def prep_sqlite():
    db_url = (
        "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    )

    # The backup lets us restart for each tutorial section
    # backup_file = "./data/travel2.backup.sqlite"
    overwrite = False
    if overwrite or not os.path.exists(DB_FILE_PATH):
        response = requests.get(db_url)
        response.raise_for_status()  # Ensure the request was successful
        with open(DB_FILE_PATH, "wb") as f:
            f.write(response.content)
        # Backup - we will use this to "reset" our DB in each section
        # shutil.copy(local_file, backup_file)
    # Convert the flights to present time for our tutorial
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    db = DB_FILE_PATH

    for table_name, df in tdf.items():
        print(f">>> {table_name}")
        print(df)


def prep_txt():
    response = requests.get(
        "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
    )
    txt_file_path = "./data/swiss_faq.md"

    response.raise_for_status()
    faq_text = response.text
    with open(txt_file_path, "a") as file:
        file.write(faq_text)


if __name__ == "__main__":
    prep_sqlite()
    # prep_txt()
