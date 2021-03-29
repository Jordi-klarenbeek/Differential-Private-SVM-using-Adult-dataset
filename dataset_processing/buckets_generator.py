import math
from datetime import datetime
import pandas as pd
import numpy as np

def generate_random_index_buckets(df):
    # Randomize the order of the data set
    df_random = df.sample(frac=1)

    # Define the fold we want to use for cross-validation
    cv = 10

    # Get the number of rows in the data set
    n = df_random.shape[0]

    # Compute the bucket size
    bucket_size = math.floor(n / cv)

    # The first bucket will be slightly larger than the rest, because it will store the residu as well
    left_over = n % cv

    # Initialize a list to store the generated buckets in
    buckets = []

    # Generate the buckets
    for i in range(1, (cv + 1)):
        if i == cv:
            # Also include all the left over records in the last bucket
            bucket = df_random.iloc[(i - 1) * bucket_size:].index.values
        else:
            bucket = df_random.iloc[(i - 1) * bucket_size: i * bucket_size].index.values

        # Append the bucket to the buckets lists
        buckets.append(bucket)

    # Convert the list to a DataFrame
    buckets_df = pd.DataFrame(buckets)

    # Check for duplicates (note that these should not exist either way, but if they do, we will try again)
    n_duplicates = buckets_df.duplicated().sum()

    if n_duplicates > 0:
        print("Duplicate indices were found unexpectedly. Let's try again!")
        return generate_random_index_buckets(df)
    else:
        print("No duplicate indices were found!")

    filename = 'buckets_' + datetime.today().strftime('%Y_%m_%d-%H_%M_%S') + '.csv'

    # If no duplicates were found, export the buckets as CSV file
    buckets_df.to_csv(filename, index=False, header=True)

    return filename