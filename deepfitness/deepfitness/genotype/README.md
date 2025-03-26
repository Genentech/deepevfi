# genotype

This folder contains functions for processing genotype strings, as a column
in a CSV, into featurized torch tensors.

Genotypes are expected to be valid according to a schema (`schemas.py`).
Featurizers (`featurizers.py`) convert genotype strings matching a schema
into a torch tensor object.
For standard training, featurized genotype tensors are coupled with
fields `count` (in a round) and `next_count` (in the next round), to
learn the sequence-to-fitness function from evolutionary dynamics.
The object type `TSNGSDatapoint` holds these three fields.

Batching is performed by the torch DataLoader, which calls a collate function
(`collaters.py`) on a list of TSNGSDatapoints, to return a TSNGSDataBatch
object. This object is used by the DeepFitnessModel(LightningModule) for 
training.

A `dataflow` is a collection of a schema, featurizer, and collater.
Example dataflows include:
- `string_to_tensor`
- `HELM_to_MF`
- `HELM_to_atomgraph`

### Example: string_to_tensor
The dataflow string_to_tensor uses a loose genotype schema that accepts any
string. In general, this dataflow supports variable-length strings.
Its featurizer computes the observed alphabet, then performs one-hot-encoding.
The collater function pads sequences to the max length, and equips the 
output TSNGSDataBatch with the padding mask for the model to use.