"""
    Featurizers are stateful classes with a featurize() function
    that converts str -> torch.tensor.
    Any state (e.g., alphabet) should be specified in args.
"""
import functools
import torch
import numpy as np

from rdkit.Chem.AllChem import MolFromSmiles, GetHashedMorganFingerprint

from hackerargs import args
from deepfitness.genotype import converters


alphabets = {
    'nucleotides': list('ACGT'),
    'amino acids': list('ACDEFGHIKLMNPQRSTVWY'),
}


class GenotypeFeaturizer:
    def __init__(self):
        pass
    
    def featurize(self, x: str) -> torch.Tensor:
        """ Converts x (str) into a torch tensor.
            Can be a converter or a chain of converters.
            Batched version unnecessary, as dataloader handles that.
        """
        raise NotImplementedError


class StringToTensorFeaturizer(GenotypeFeaturizer):
    def __init__(self):
        """ Featurizes variable-length strings into variable-length tensors.
            Use with pad_variable_len collater, with transformers.
        """
        self.args = {
            'alphabet': args.get('ft.alphabet'),
        }
        self.alphabet = alphabets[args.get('ft.alphabet')]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def __repr__(self) -> str:
        """ String representation with settings.
            Used to specify featurizer in feature store.
        """
        return 'str-to-tensor' + '_'.join([f'{k}-{v}' for k, v in self.args.items()])
    
    @functools.cache
    def featurize(self, x: str) -> torch.Tensor:
        return converters.str2idxtensor(x, self.char_to_idx)


class StringToOneHotFeaturizer(GenotypeFeaturizer):
    def __init__(self):
        """ Featurizes variable-length strings into fixed-length (`len`)
            tensors, with padding. Use with default collater, with MLP.
        """
        self.args = {
            'alphabet': args.get('ft.alphabet'),
            'len': args.get('ft.onehot_len')
        }
        self.alphabet = alphabets[args.get('ft.alphabet')]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        padding_chars = list('!@#$%^&*()-=_+[];:/?<,>.')
        pc_cands = [c for c in padding_chars if c not in self.alphabet]
        assert len(pc_cands) > 0, 'Failed to find a valid padding character'
        self.padding_char = pc_cands[0]
        self.char_to_idx[self.padding_char] = len(self.alphabet)
        if '-' not in self.alphabet:
            self.char_to_idx['-'] = len(self.alphabet)
        self.n_classes = len(set(self.char_to_idx.values()))

    def __repr__(self) -> str:
        """ String representation with settings.
            Used to specify featurizer in feature store.
        """
        return 'str-to-onehot' + '_'.join([f'{k}-{v}' for k, v in self.args.items()])
    
    @functools.cache
    def featurize(self, x: str) -> torch.Tensor:
        assert len(x) <= self.args['len'], f'{len(x)=} is longer than one-hot length'
        idx_tensor = converters.str2idxtensor(
            x + self.padding_char * (self.args['len'] - len(x)), 
            self.char_to_idx
        )
        oh = torch.functional.F.one_hot(idx_tensor, num_classes = self.n_classes)
        ft = torch.flatten(oh)
        return ft.to(torch.float)


class SmilesToFingerprintFeaturizer(GenotypeFeaturizer):
    def __init__(self):
        self.args = {
            'radius': args.setdefault('ft.mfp_radius', 3),
            'n_bits': args.setdefault('ft.mfp_nbits', 128),
            'use_chirality': args.setdefault('ft.use_chirality', True)
        }

    def __repr__(self) -> str:
        """ String representation with settings.
            Used to specify featurizer in feature store.
        """
        return 'smi-to-mfp_' + '_'.join([f'{k}-{v}' for k, v in self.args.items()])

    @functools.cache
    def featurize(self, smiles: str) -> torch.Tensor:
        """ Featurize SMILES into morgan fingerprint.
            
            GetHashedMorganFingerprint with useChirality = False is equivalent to count fingerprint, produced using
            >>> MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius, fpSize = nBits)
            >>> MFPGEN.GetCountFingerprint(mol)
        """
        mol = MolFromSmiles(smiles)
        mfp = GetHashedMorganFingerprint(
            mol,
            radius = self.args['radius'],
            nBits = self.args['n_bits'],
            useChirality = self.args['use_chirality'],
        )
        return torch.tensor(np.array(list(mfp)), dtype = torch.float32)


# getter
def get_featurizer(name: str) -> GenotypeFeaturizer:
    name_to_featurizer = {
        'string_to_tensor': StringToTensorFeaturizer,
        'smiles_to_fingerprint': SmilesToFingerprintFeaturizer,
    }
    return name_to_featurizer[name]