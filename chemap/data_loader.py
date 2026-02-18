import os
import pathlib
import re
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

    def load_collection(self, source: str, **kwargs) -> list:
        """
        Loads a dataset collection from a DOI-based registry (e.g. Zenodo).

        Parameters
        -------------
        source:
            A DOI.

        Returns
        -------------
        list of downloaded filenames from the registry.

        Raises
        -------------
        ValueError if DOI not present.
        """
        doi_pattern = r'(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)'

        if not source.startswith("doi") or not bool(re.search(doi_pattern, source)):
            ValueError(f"Could not detect DOI in source {source}.")

        return self._from_registry(source, **kwargs)

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

        column_map = {col.lower(): col for col in df.columns}
        target_col = column_map.get(smiles_column.lower())

        if not target_col:
            raise ValueError(f"Smiles column {smiles_column} not in dataframe.")

        return df[target_col].tolist()

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

    def _from_registry(self, doi: str, **kwargs) -> list:
        """
        Loads a dataset collection from DOI-based registry (e.g., Zenodo).

        Parameters
        -------------
        doi:
            A valid DOI string.

        Returns
        -------------
        list of strings with absolute path for all downloaded files.

        Raises
        -------------
        ValueError if file type unsupported.
        ValueError if smiles column not present.
        """
        if not doi.startswith("doi"):
            doi = f"doi:{doi}"

        client = pooch.create(
            path=self.cache_dir,
            base_url=f"{doi}/",
            registry=None,
        )
        client.load_registry_from_doi()

        return [client.fetch(f, progressbar=True) for f in client.registry]