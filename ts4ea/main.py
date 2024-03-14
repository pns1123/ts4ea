from fastapi import FastAPI
import random

app = FastAPI()

# Function to get a random image using the provided algorithm
def get_random_image():
    # For example, selecting a random image from a list
    image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", "/path/to/image3.jpg"]
    random_image_path = random.choice(image_paths)
    return random_image_path

# Endpoint to fetch a random image
@app.get("/api/images/random")
async def get_random_image_endpoint():
    image_path = get_random_image()
    return {"image_path": image_path}

# Endpoint to submit user response
@app.post("/api/images/{image_id}/responses")
async def submit_user_response(image_id: int, user_agreement: bool, explanation_helpful: bool):
    return {"message": "User response submitted successfully"}
