from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/items.csv")
df.head()


# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time:
          1. Import the datetime module.
          2. Create a datetime object for your desired time.
          3. Use uuid_from_time(your_datetime) instead of uuid_from_time(datetime.now()).

        Example:
            from datetime import datetime
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid_from_time(specific_time))

        This is useful when your content already has an associated datetime.
    """
    product = row["PRODUCT_NAME"]
    content = row["PRODUCT_DESCRIPTION"]
    embedding = vec.get_embedding(content)

    def safe_value(val):
        return None if pd.isna(val) else val

    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "product": product,
                "category-1": safe_value(row["CATEGORY_L1"]),
                "category-2": safe_value(row["CATEGORY_L2"]),
                "price": safe_value(row["PRICE"]),
                "gender": safe_value(row["GENDER"]),
                "promoted": safe_value(row["PROMOTED"]),
                "created_at": datetime.now().isoformat(),
            },
            "contents": f"{product} - {content}",
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

# def clean_all_nans(metadata):
#     if isinstance(metadata, dict):
#         return {
#             k: (None if isinstance(v, float) and math.isnan(v) else v)
#             for k, v in metadata.items()
#         }
#     return metadata

# records_df["metadata"] = records_df["metadata"].apply(clean_all_nans)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskANNIndex
vec.create_keyword_search_index()  # GIN Index
vec.upsert(records_df)
