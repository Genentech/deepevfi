
method_standard_naming = {
    'acides': 'ACIDES',
    'enrich2': 'Enrich2',
    'sfit-mul': 'EVFI',
    'deep_latent': 'DeepEVFI',
}


def rename_method(method: str) -> str:
    return method_standard_naming.get(method, method)


dataset_standard_naming = {
    'A': 'A',
    'B': 'B',
    'C': 'C',
    'D': 'D',
    'E': 'E',
    'F': 'F',
    'G': 'G',
    'FGFR1-AHOent0.25': 'Y-Ab',
    'TEAD-1fc-p2tl': 'M-MP',
}


def rename_dataset(dataset: str) -> str:
    return dataset_standard_naming.get(dataset, dataset)
