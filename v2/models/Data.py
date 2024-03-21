from pydantic import BaseModel
from typing import Union

# Base class for different types of data
class Data(BaseModel):
    pass

# Subclass for text data
class TextData(Data):
    text: str

# Subclass for binary data (e.g., images)
class BinaryData(Data):
    data: bytes

# Subclass for Numerical data 
class NumericalData(Data):
    value: Union[int, float]