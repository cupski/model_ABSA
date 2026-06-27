import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    """
    Membaca berkas dataset berlabel dan mengembalikan DataFrame.

    Parameters
    ----------
    data_path : str
        Path ke berkas CSV dataset yang telah dianotasi.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom teks ('Komentar') dan kolom label aspek
        ('Content Quality', 'Subscription & Pricing', 'UI/UX',
        'Functionality', 'Technical & Access').
    """
    df = pd.read_csv(data_path)
    return df
