""" 
    Dataflows describe how genotype strings are featurized and batched
    into DataBatch objects, provided to a model.
    A dataflow is a named collection of a genotype schema, featurizer, and 
    collater. The schema describes what valid genotype strings are in a CSV.
    Featurizer converts strings to DataPoints, 
    and collater packs a list of DataPoints into a TSNGSDataBatch.
"""

from dataclasses import dataclass
from deepfitness.genotype import schemas, featurizers, collaters
from deepfitness.genotype.feature_stores import FeatureStore


@dataclass
class DataFlow:
    schema: schemas.GenotypeStrSchema
    featurizer: featurizers.GenotypeFeaturizer
    collater: callable
    feature_store: FeatureStore | None = None

    def add_feature_store(self, feature_store: FeatureStore):
        self.feature_store = feature_store


# getter
def get_dataflow(name: str) -> DataFlow:
    if name == 'string_to_tensor':
        return DataFlow(
            schema = schemas.AnyStringSchema(),
            featurizer = featurizers.StringToTensorFeaturizer(),
            collater = collaters.pad_variable_len_seqs,
        )
    if name == 'string_to_onehot':
        return DataFlow(
            schema = schemas.AnyStringSchema(),
            featurizer = featurizers.StringToOneHotFeaturizer(),
            collater = collaters.default_collater,
        )
    if name == 'smiles_to_fingerprint':
        return DataFlow(
            schema = schemas.SMILESSchema(),
            featurizer = featurizers.SmilesToFingerprintFeaturizer(),
            collater = collaters.default_collater,
        )
    assert False, 'Invalid dataflow name.'