# Atharva Mahule - Personal Portfolio

[![Live Demo](https://img.shields.io/badge/Live%20Demo-View%20Site-blue?style=for-the-badge)](https://portfolio-6-l6vf.onrender.com/)

An interactive and dynamic personal portfolio showcasing my projects, skills, and experience in Machine Learning, Artificial Intelligence, and Web Development. This repository contains the source code for the website, which goes beyond a standard static page by featuring a custom-trained NLP chatbot integrated directly into the application to engage visitors and answer questions.

##  Features

* **Interactive AI Chatbot:** A custom Natural Language Processing (NLP) assistant trained to answer queries about my background, skills, and projects.
* **Modern Web Interface:** A fully responsive frontend built with HTML, CSS, and JavaScript.
* **Robust Backend:** Powered by Python and Flask to handle routing, NLP model inference, and seamless server-side operations.
* **Cloud Ready:** Fully configured for deployment on Render.

##  Tech Stack

* **Languages:** Python, JavaScript, HTML5, CSS3
* **Backend Framework:** Flask
* **Machine Learning / NLP:** NLTK (Natural Language Toolkit)
* **Deployment:** Render (`Procfile`, `build.sh`)

##  Repository Structure

* `app.py`: The main Flask application script handling routes, API endpoints, and chatbot inference.
* `train.py`: The machine learning script used to process data and train the NLP model.
* `intro.json`: The dataset and conversational intents used to train the chatbot.
* `connect.py`: Handles database or external service connections.
* `upload.py`: Utility script for handling data/file uploads.
* `models/`: Directory containing the compiled and saved machine learning models.
* `nltk_data/`: Tokenizers and necessary linguistic data for NLTK operations.
* `templates/`: HTML templates for the frontend UI.
* `static/`: CSS stylesheets, client-side JavaScript, and image assets.
* `Procfile` & `build.sh`: Configuration files essential for the Render deployment pipeline.
* `requirements.txt`: Complete list of Python dependencies required to run the environment.

##  Local Installation & Setup

To run this project locally on your machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AtharvaM25/Portfolio.git](https://github.com/AtharvaM25/Portfolio.git)
    cd Portfolio
    ```

2.  **Create a virtual environment (optional but highly recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the chatbot model:**
    ```bash
    python train.py
    ```

5.  **Run the web application:**
    ```bash
    python app.py
    ```

6.  **View the site:**
    Open your web browser and navigate to `http://127.0.0.1:5000` (or the port specified in your terminal output).
