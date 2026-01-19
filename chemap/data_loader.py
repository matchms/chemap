import os
import pathlib
import pandas as pd
import pooch


class DatasetLoader:
    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = cache_dir

    def load(self, source: str, **kwargs) -> list:
        """
        Loads a dataset from local file or web.

        Parameters
        -------------
        source:
            Either load a local file or a hyperlink pointing to a remote file.
            Supported filetypes: .csv, .json, .parquet, .xls, .xlsx, .xlsx.

        Returns
        -------------
        list of smiles strings.

        Raises
        -------------
        ValueError if neither local file nor http/ftp/sftp.
        """
        if os.path.exists(source):
            return self._from_local_file(source, **kwargs)
        elif source.startswith(("http", "ftp", "sftp")):
            return self._from_web(source, **kwargs)
        else:
            raise ValueError(f"Source {source} unknown.")

    def _from_local_file(self, path, smiles_column: str = "smiles") -> list:
        """
        Loads a dataset from local file.

        Parameters
        -------------
        path:
            string of local file path.

        smiles_column:
            Name of column containing smiles. Defaults to smiles

        Returns
        -------------
        list of smiles strings.

        Raises
        -------------
        ValueError if file type unsupported.
        ValueError if smiles column not present.
        """
        suffix = pathlib.Path(path).suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        elif suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Fileformat {suffix} not supported.")

        if smiles_column not in df.columns:
            raise ValueError(f"Smiles column {smiles_column} not in dataframe.")

        return df[smiles_column].tolist()

    def _from_web(self, url: str, **kwargs) -> list:
        """
        Loads a dataset from web.

        Parameters
        -------------
        url:
            string of url.

        Returns
        -------------
        list of smiles strings.
        """
        file_path = pooch.retrieve(
            url=url,
            known_hash=kwargs.get("hash", None),
            path=self.cache_dir,
            progressbar=True,
        )

        return self._from_local_file(file_path, **kwargs)
