from pydantic import BaseModel
from typing import Union

# import Data class
from .Data import Data

# Explanation class
class Explanation(BaseModel):
    data_point: Data  # The original media associated with the explanation
    explanation_content: Data  # The content of the explanation
    prediction: Data  # The prediction associated with the explanation
    ground_truth: Data  # The ground truth associated with the explanation
    metadata: dict = {}  # Optional extra data field associated with the explanation