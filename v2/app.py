from fastapi import FastAPI
from typing import List, Optional
from pydantic import Field
import uuid

# Import Pydantic models
from models.UserFeedback import DiscreteResponse, ContinuousResponse

# Create FastAPI instance
app = FastAPI()

# dummy list of explanations
explanations = [
    {
        "id": 1,
        "title": "Explanation 1",
        "content": "This is explanation 1"
    },
    {
        "id": 2,
        "title": "Explanation 2",
        "content": "This is explanation 2"
    }
]

# Initialize reward model and start loop internally
def initialize_reward_model():
    # Your initialization logic here
    # For example, you could initialize the reward model and start the system loop
    print("Reward model initialized and system loop started successfully")


# API endpoint to handle session start request
@app.post("/session/start")
async def start_session():
    # Assign a unique ID to the user
    user_id = str(uuid.uuid4())
    
    # Initialize reward model and start loop
    initialize_reward_model()

    return {"user_id": user_id, "Generated Explanation": explanations[0], "message": "Session started successfully"}   


# Endpoint to receive user feedback
@app.post("/feedback")
async def receive_feedback(discrete_response: Optional[DiscreteResponse] = None, continuous_response: Optional[ContinuousResponse] = None):
    if discrete_response:
        # Process discrete response
        print("Received discrete response:", discrete_response)
    elif continuous_response:
        # Process continuous response
        print("Received continuous response:", continuous_response)
    else:
        return {"message": "No feedback received"}

    return {"discrete_response": discrete_response , "continous_response": continuous_response, "Next Generated Explanation": explanations[1], "message": "Feedback received successfully"}






