class PymmunomicsBaseWarning(Warning):
    pass

class ArgumentWarning(PymmunomicsBaseWarning):
    pass

class AmbiguousValuesWarning(PymmunomicsBaseWarning):
    pass

class PymmunomicsBaseError(Exception):
    pass

class InvalidArgumentError(PymmunomicsBaseError):
    pass
