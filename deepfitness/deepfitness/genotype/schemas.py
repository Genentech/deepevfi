import functools
from rdkit import Chem


class GenotypeStrSchema:
    """ Parent class for schema for genotype strings in CSV. """
    def __init__(self):
        pass
    
    def is_valid(self, x: str) -> bool:
        raise NotImplementedError
    

class AnyStringSchema(GenotypeStrSchema):
    """ Accepts any string """
    def __init__(self):
        pass
    
    def is_valid(self, x: str) -> bool:
        return type(x) == str
    

class SMILESSchema(GenotypeStrSchema):
    """ Accepts SMILES strings. """
    def __init__(self):
        pass

    @functools.cache
    def is_valid(self, x: str) -> bool:
        mol = Chem.MolFromSmiles(x, sanitize = False)
        return mol is not None
        

# getter
def get_schema(name: str) -> GenotypeStrSchema:
    name_to_schema = {
        'AnyStringSchema': AnyStringSchema(),
        'SMILESSchema': SMILESSchema(),
    }
    return name_to_schema[name]