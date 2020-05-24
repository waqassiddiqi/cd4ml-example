import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    try:
        df = pd.read_csv(csv_url, sep=';')
        
        msk = np.random.rand(len(df)) < 0.9
        df_test_train = df[msk]
        df_validation = df[~msk]
        
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV. Error: %s", e)
    
    df_test_train.to_csv("data/output.csv")
    df_validation.to_csv("data/output_validate.csv")