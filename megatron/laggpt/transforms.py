import numpy as np

from gluonts.transform import (
    SimpleTransformation, 
    ExpandDimArray,
    AddObservedValuesIndicator,
    SetField,
    TargetDimIndicator,
    Identity,
    InstanceSplitter
)
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import DataEntry
from hashlib import sha1
from typing import Type, List, Optional

class HashField(SimpleTransformation):

    def __init__(
        self, 
        field_name: str, 
        hash_alg: Optional[Type] = sha1, 
        dtype: Type = np.float32
    ):
        self.field_name = field_name
        self.hash_alg = hash_alg
        self.dtype = dtype

    def transform(self, data: DataEntry) -> DataEntry:
        value = data[self.field_name]
        hash = self.hash_alg(str(value).encode("utf-8"))
        data[self.field_name] = np.array([int(hash.hexdigest()[:8], 16)]).astype(self.dtype, copy=False)
        return data

class SqueezeDimArray(SimpleTransformation):

    def __init__(self, field: str, axis: int):
        self.field = field
        self.axis = axis

    def transform(self, data: DataEntry) -> DataEntry:
        for field in [
            f"past_{self.field}",
            f"future_{self.field}",
            f"past_{FieldName.OBSERVED_VALUES}",
            f"future_{FieldName.OBSERVED_VALUES}",
        ]:
            data[field] = np.squeeze(data[field], axis=self.axis)
        return data

from gluonts.transform import AsNumpyArray, SetFieldIfNotPresent
def get_transformation(is_multi_target: Optional[bool] = False, dataset_id: Optional[int] = None):
    transformation = AsNumpyArray(FieldName.TARGET, expected_ndim=2 if is_multi_target else 1)
    if not is_multi_target:
        # Ensure that 'target' is 2D array for target sampling
        transformation += ExpandDimArray(field=FieldName.TARGET, axis=0)

    # Replace NaNs with zeros
    transformation += AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )
    transformation += TargetDimIndicator(field_name=FieldName.TARGET + '_dim', target_field=FieldName.TARGET)

    # transformation += SetFieldIfNotPresent(field=FieldName.START, value=0)
    transformation += SetField(output_field=FieldName.START, value=0)
    transformation += HashField(field_name=FieldName.ITEM_ID)
    transformation += SetField(output_field="dataset_id", value=np.float32([dataset_id]) if dataset_id is not None else -1)

    return transformation

def get_transformation_gluonts(is_multi_target: Optional[bool] = False):
    transformation = Identity()
    if not is_multi_target:
        # Ensure that 'target' is 2D array for target sampling
        transformation += ExpandDimArray(field=FieldName.TARGET, axis=0)
    
    # Replace NaNs with zeros
    transformation += AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )
    transformation += SetField(output_field=FieldName.START, value=0)
    transformation += TargetDimIndicator(field_name=FieldName.TARGET + '_dim', target_field=FieldName.TARGET)

    return transformation

class MetadataInstanceSplitter(InstanceSplitter):

    def __init__(self, meta_fields: List[str] = [], *args, **kwargs):
        self.meta_fields = meta_fields
        super().__init__(*args, **kwargs)

    def _split_instance(self, entry: DataEntry, idx: int) -> DataEntry:
        split_entry = super()._split_instance(entry, idx)

        for field in self.meta_fields:
            split_entry[field] = entry[field]
        return split_entry   
