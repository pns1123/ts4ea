const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();

// Define the directory where streetview images are stored
const streetviewDirectory = "streetview";

// Function to get a random image from the streetview directory
 function getRandomImage() {
//     const imageFiles = fs.readdirSync(streetviewDirectory).filter(file => fs.statSync(path.join(streetviewDirectory, file)).isFile());
//     const randomImage = imageFiles[Math.floor(Math.random() * imageFiles.length)];
//     return path.join(streetviewDirectory, randomImage);
}

// Endpoint to fetch a random image
app.get("/api/images/random", (req, res) => {
    const imagePath = getRandomImage();
    res.json({ image_path: imagePath });
});

// Endpoint to submit user response
app.post("/api/images/:imageId/responses", (req, res) => {
    // const imageId = req.params.imageId;
    // const userAgreement = req.body.userAgreement;
    // const explanationHelpful = req.body.explanationHelpful;
    // res.json({ message: "User response submitted successfully" });
});
