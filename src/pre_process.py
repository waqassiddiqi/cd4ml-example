import pandas as pd

if __name__ == "__main__":
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    try:
        df = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV. Error: %s", e)
    
    df.to_csv("data/output.csv")