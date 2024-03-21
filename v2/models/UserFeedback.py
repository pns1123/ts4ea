from pydantic import BaseModel, validator
from typing import List

class DiscreteResponse(BaseModel):
    options: List[str]
    user_choice: str

class ContinuousResponse(BaseModel):
    min: float
    max: float
    user_choice: float 

    # Validator to check if user choice is between min and max
    @validator('user_choice')
    def check_user_choice(cls, v, values):
        if v is not None and ('min' in values and 'max' in values):
            if not values['min'] <= v <= values['max']:
                raise ValueError("User choice must be between min and max")
        return v

    
