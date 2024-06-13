from gluonts.dataset.repository.datasets import get_dataset as get_gluonts_dataset
import random
import numpy as np
from .lotsa_dataloaders import (
    load_lotsa_arrow_dataset, load_gluonts_eval_dataset
)
from typing import List, Literal, Optional, Iterable

from pathlib import Path

from tempfile import TemporaryDirectory
from zipfile import ZipFile
from functools import partial

import gluonts
from gluonts.dataset.arrow import ArrowStreamFile
from gluonts.dataset.split import TestData, split
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets, DatasetCollection
from gluonts.dataset.repository._tsf_datasets import Dataset as MonashDataset
from gluonts.dataset.repository._tsf_datasets import TSFReader, convert_data
from gluonts.dataset.repository._tsf_reader import frequency_converter
from gluonts.dataset.repository._util import metadata
from gluonts.dataset.repository.datasets import get_dataset
from pandas.tseries.frequencies import to_offset

from gluonts.transform import SetField
from .transforms import get_transformation
from .lotsa_dataset_names import IN_DISTR_EVAL

import yaml
import numpy as np
import json

from typing import Optional, Literal

def default_prediction_length_from_frequency(freq: str) -> int:
    prediction_length_map = {
        "T": 60 * 24 * 7,
        "H": 24 * 7,
        "D": 30,
        "W-SUN": 8,
        "M": 12,
        "Y": 4,
        "S": 60 * 60 * 24 * 7,
    }
    try:
        freq = to_offset(freq).name
        return prediction_length_map[freq]
    except KeyError as err:
        raise ValueError(
            f"Cannot obtain default prediction length from frequency `{freq}`."
        ) from err


gluonts.dataset.repository._tsf_datasets.default_prediction_length_from_frequency = (
    default_prediction_length_from_frequency
)


def generate_forecasting_dataset(
    dataset_path: Path,
    dataset_name: str,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
):
    dataset = gluonts.dataset.repository._tsf_datasets.datasets[dataset_name]
    dataset_path.mkdir(exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        with ZipFile(dataset.download(temp_path)) as archive:
            archive.extractall(path=temp_path)

        # only one file is exptected
        reader = TSFReader(temp_path / archive.namelist()[0])
        meta, data = reader.read()

    if dataset_name.startswith("cif_2016") and len(dataset_name) > len("cif_2016"):
        horizon = int(dataset_name[len("cif_2016_") :])
        data = list(filter(lambda x: x if x["horizon"] == horizon else False, data))
        meta.forecast_horizon = horizon

    if dataset_name.startswith("monash_m3_other"):
        meta.frequency = "quarterly"

    freq = frequency_converter(meta.frequency)
    if prediction_length is None:
        if hasattr(meta, "forecast_horizon"):
            prediction_length = int(meta.forecast_horizon)
        else:
            prediction_length = default_prediction_length_from_frequency(freq)

    # Impute missing start dates with unix epoch and remove time series whose
    # length is less than or equal to the prediction length
    data = [
        {**d, "start_timestamp": d.get("start_timestamp", "1970-01-01")}
        for d in data
        if len(d[FieldName.TARGET]) > prediction_length
    ]
    train_data, test_data = convert_data(data, prediction_length)

    meta = MetaData(
        **metadata(
            cardinality=len(data),
            freq=freq,
            prediction_length=prediction_length,
        )
    )

    dataset = TrainDatasets(metadata=meta, train=train_data, test=test_data)
    dataset.save(path_str=str(dataset_path), writer=dataset_writer, overwrite=True)


gluonts.dataset.repository._tsf_datasets.generate_forecasting_dataset = (
    generate_forecasting_dataset
)


additional_datasets = {
    "bitcoin": MonashDataset(
        file_name="bitcoin_dataset_without_missing_values.zip",
        record="5122101",
        ROOT="https://zenodo.org/record",
    ),
    "wind_power": MonashDataset(
        file_name="wind_4_seconds_dataset.zip",
        record="4656032",
        ROOT="https://zenodo.org/record",
    ),
    "us_births": MonashDataset(
        file_name="us_births_dataset.zip",
        record="4656049",
        ROOT="https://zenodo.org/record",
    ),
    "traffic_hourly": MonashDataset(
        file_name="traffic_hourly_dataset.zip",
        record="4656132",
        ROOT="https://zenodo.org/record",
    ),
    "traffic_weekly": MonashDataset(
        file_name="traffic_weekly_dataset.zip",
        record="4656135",
        ROOT="https://zenodo.org/record",
    ),
    "solar_power": MonashDataset(
        file_name="solar_4_seconds_dataset.zip",
        record="4656027",
        ROOT="https://zenodo.org/record",
    ),
    "oikolab_weather": MonashDataset(
        file_name="oikolab_weather_dataset.zip",
        record="5184708",
        ROOT="https://zenodo.org/record",
    ),
    "elecdemand": MonashDataset(
        file_name="elecdemand_dataset.zip",
        record="4656069",
        ROOT="https://zenodo.org/record",
    ),
    "covid_mobility": MonashDataset(
        file_name="covid_mobility_dataset_with_missing_values.zip",
        record="4663762",
        ROOT="https://zenodo.org/record",
    ),
    "extended_web_traffic_with_missing": MonashDataset(
        file_name="web_traffic_extended_dataset_with_missing_values.zip",
        record="7370977",
        ROOT="https://zenodo.org/record",
    ),
    "monash_m3_monthly": MonashDataset(
        file_name="m3_monthly_dataset.zip",
        record="4656298",
        ROOT="https://zenodo.org/record",
    ),
    "monash_m3_quarterly": MonashDataset(
        file_name="m3_quarterly_dataset.zip",
        record="4656262",
        ROOT="https://zenodo.org/record",
    ),
    "monash_m3_yearly": MonashDataset(
        file_name="m3_yearly_dataset.zip",
        record="4656222",
        ROOT="https://zenodo.org/record",
    ),
    "monash_m3_other": MonashDataset(
        file_name="m3_other_dataset.zip",
        record="4656335",
        ROOT="https://zenodo.org/record",
    ),
    "cif_2016_12": MonashDataset(
        file_name="cif_2016_dataset.zip",
        record="4656042",
        ROOT="https://zenodo.org/record",
    ),
    "cif_2016_6": MonashDataset(
        file_name="cif_2016_dataset.zip",
        record="4656042",
        ROOT="https://zenodo.org/record",
    ),
    "sunspot_with_missing": MonashDataset(
        file_name="sunspot_dataset_with_missing_values.zip",
        record="4654773",
        ROOT="https://zenodo.org/record",
    ),
    "temperature_rain_with_missing": MonashDataset(
        file_name="temperature_rain_dataset_with_missing_values.zip",
        record="5129073",
        ROOT="https://zenodo.org/record",
    ),
    "rideshare_with_missing": MonashDataset(
        file_name="rideshare_dataset_with_missing_values.zip",
        record="5122114",
        ROOT="https://zenodo.org/record",
    ),
    "car_parts_with_missing": MonashDataset(
        file_name="car_parts_dataset_with_missing_values.zip",
        record="4656022",
        ROOT="https://zenodo.org/record",
    ),
    "kdd_cup_2018_with_missing": MonashDataset(
        file_name="kdd_cup_2018_dataset_with_missing_values.zip",
        record="4656719",
        ROOT="https://zenodo.org/record",
    ),
    "vehicle_trips_with_missing": MonashDataset(
        file_name="vehicle_trips_dataset_with_missing_values.zip",
        record="5122535",
        ROOT="https://zenodo.org/record",
    ),
    "bitcoin_with_missing": MonashDataset(
        file_name="bitcoin_dataset_with_missing_values.zip",
        record="5121965",
        ROOT="https://zenodo.org/record",
    ),
    "london_smart_meters_with_missing": MonashDataset(
        file_name="london_smart_meters_dataset_with_missing_values.zip",
        record="4656072",
        ROOT="https://zenodo.org/record",
    ),
    "wind_farms_with_missing": MonashDataset(
        file_name="wind_farms_minutely_dataset_with_missing_values.zip",
        record="4654909",
        ROOT="https://zenodo.org/record",
    ),
    "nn5_daily_with_missing": MonashDataset(
        file_name="nn5_daily_dataset_with_missing_values.zip",
        record="4656110",
        ROOT="https://zenodo.org/record",
    ),
}

gluonts.dataset.repository._tsf_datasets.datasets.update(additional_datasets)
gluonts.dataset.repository.datasets.dataset_recipes.update({
    k: partial(
        generate_forecasting_dataset,
        dataset_name=k,
    )
    for k in additional_datasets.keys()
})


import random
class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights.copy()
        self._rng = random.Random(seed)

    def __next__(self):
        
        data = None
        while not data:
            (index, ) = self._rng.choices(range(len(self._datasets)), weights=self._weights, k=1)
            try:
                data = next(self._datasets[index])
            except StopIteration:
                del self._datasets[index]
                del self._weights[index]
            if len(self._datasets) == 0:
                raise StopIteration
        
        return data
    
class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)
    
    def __len__(self):
        return sum([len(ds) for ds in self._datasets])

from gluonts.transform import TransformedDataset, Transformation, Chain, MapTransformation, FlatMapTransformation
from gluonts.dataset import DataEntry, Dataset
from typing import Optional, Iterator
import random

class ShuffledTransformedDataset(TransformedDataset):
    def __init__(
            self, 
            dataset: Dataset, 
            transformation: Transformation,
            is_train: Optional[bool] = True,
            seed: Optional[int] = None
        ):
        self.base_dataset = dataset
        self.transformation = transformation
        self.is_train = is_train

        if not isinstance(self.transformation, Chain):
            self.transformation = Chain([self.transformation])

        self.rng = random.Random(seed)
        self.shuffle_indices = self.rng.sample(
            range(len(self.base_dataset)),
            k=len(self.base_dataset)
        )

    def transform_entry(self, data: DataEntry) -> DataEntry:
        for t in self.transformation.transformations:
            if isinstance(t, MapTransformation):
                data = t.map_transform(data, self.is_train)
            elif isinstance(t, FlatMapTransformation):
                data = t.flatmap_transform(data, self.is_train)
            else:
                raise RuntimeError(f"Expected transfromation {t} to be of type MapTransformation or FlatMapTransformation but got {type(t)}")

        return data
    
    def __iter__(self) -> Iterator[DataEntry]:
        for idx in self.shuffle_indices:
            yield self.transform_entry(self.base_dataset[idx])


def load_train_dataset(path : Path, seed : int): 
    datasets = []
    loaded = []
    weights = []
    with open("/lustre/orion/csc605/world-shared/cross-modal-transfer/lotsa_capped_weights.yml", "r") as infile:
        lotsa_weights = yaml.safe_load(infile)
    
    for k, dataset_path in enumerate(path.glob("[!.]*")): # skip hidden directories
        if dataset_path.is_dir():
            ds = load_lotsa_arrow_dataset(dataset_path.name, dataset_path.parent, dataset_id=k)
            loaded.append(dataset_path.name)
            datasets.append(ds)
            weights.append(lotsa_weights[dataset_path.name])

    print(f"Loading {loaded} ...")
    return CombinedDataset(datasets=datasets, seed=seed, weights=weights) # TODO: Add options for using uniform weights or DatasetCollection

from gluonts.itertools import RandomYield
def load_lotsa_train_dataset(dataset_name : str, path : Path, dataset_id: Optional[int] = None) -> Dataset:
    # Keep signature like this in case we want to load individual datasets
    dataset_path = path / dataset_name

    info_path = dataset_path / "dataset_info.json"
    with open(info_path, "r") as info_file: 
        metadata = json.load(info_file)
        is_multi_target = 'length' in metadata["features"]["target"] and metadata["features"]["target"]['length'] > 1
    
    transformation = get_transformation(is_multi_target)
    if dataset_id is not None:
        transformation += SetField(output_field="dataset_id", value=np.float32([dataset_id]))

    dataset_shards = [transformation.apply(ArrowStreamFile(path=arrow_path)) for arrow_path in dataset_path.glob("*.arrow")]
    dataset = RandomYield(iterables=dataset_shards) # TODO: Add option to use DatasetCollection
    return dataset

from datasets import load_from_disk
def load_lotsa_arrow_dataset(
    dataset_name : str, 
    path : Path, 
    dataset_id: Optional[int] = None,
    seed: Optional[int] = 1337
    ) -> Dataset:
    dataset_path = path / dataset_name

    info_path = dataset_path / "dataset_info.json"
    with open(info_path, "r") as info_file: 
        metadata = json.load(info_file)
        is_multi_target = 'length' in metadata["features"]["target"] and metadata["features"]["target"]['length'] > 1

    transformation = get_transformation(is_multi_target, dataset_id)
    arrow_dataset = load_from_disk(dataset_path=str(dataset_path))
    return ShuffledTransformedDataset(arrow_dataset, transformation, is_train=True, seed=seed)

def load_in_distr_eval_dataset(path: Path, seed: int, split: Literal["validation", "test"], regenerate: Optional[bool] = False) -> CombinedDataset:
    datasets = [load_gluonts_eval_dataset(dataset_name, path, split, regenerate=regenerate, dataset_id=k) for k, dataset_name in enumerate(IN_DISTR_EVAL)]
    return CombinedDataset(datasets=datasets, seed=seed) # TODO: Add option to use RandomYield

def load_gluonts_eval_dataset(dataset_name: str, path: Path, eval_split: Literal["validation", "test"], regenerate: Optional[bool] = False, dataset_id: Optional[int] = None) -> TestData:
    dataset = get_dataset(
        dataset_name, 
        path=path,
        prediction_length=1, 
        regenerate=regenerate
    )

    dataset_split = dataset.train if eval_split == "validation" else dataset.test
    _, test_template = split(dataset_split, offset=-1)
    eval_data = test_template.generate_instances(prediction_length=1).dataset

    transformation = get_transformation(is_multi_target=False)
    if dataset_id is not None:
        transformation += SetField(output_field="dataset_id", value=np.float32([dataset_id]))

    eval_data = transformation.apply(eval_data)
    return eval_data



def get_combined_dataset(
        dataset_names : List[str], 
        path : Path, 
        rank : int, 
        seed : int, 
        split: Optional[Literal["train", "val", "test"]] = "train",
        weights: Optional[Iterable[float]] = None,
    ) -> CombinedDataset:
    if split == "train":
        dataset = load_train_dataset(Path("/lustre/orion/csc605/world-shared/cross-modal-transfer/salesforce-lotsa/lotsa_data"), seed=seed)
    else:
        dataset = load_in_distr_eval_dataset(Path("/lustre/orion/csc605/world-shared/cross-modal-transfer/gluonts"), seed=seed, split="validation")

    return dataset

