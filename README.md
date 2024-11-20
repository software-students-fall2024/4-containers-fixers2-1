![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)
![CI Status](https://github.com/software-students-fall2024/4-containers-fixers2-1/actions/workflows/ci.yml/badge.svg)

# Containerized App Exercise

# Table of Contents
1. [Teammates](#teammates)  
2. [App Description: Emotion Recognition and Wellness Advisor](#app-description-emotion-recognition-and-wellness-advisor)  
3. [Task Board](#task-board)
4. [What It Is](#what-it-is)
5. [How to Open](#how-to-open)

# Teammates 

[Dasha Miroshnichenko](https://github.com/dm5198)

[Emily Huang](https://github.com/emilyjhuang)

[Jessie Kim](https://github.com/jessiekim0)

[Nick Burwell](https://github.com/nickburwell)

# App Description: Emotion Recognition and Wellness Advisor

Our Emotion Recognition and Wellness Advisor app uses cutting-edge emotion detection technology to help users enhance their mental well-being. By analyzing facial expressions in real-time, the app identifies emotions such as happiness, sadness, anger, and more. Based on the detected emotion, the app provides tailored wellness advice, like mindfulness exercises, motivational quotes, or self-care suggestions.

# Task Board

[Task Board](https://github.com/orgs/software-students-fall2024/projects/105)


# What It Is

- **Emotion Detection:** Using a machine learning model, the app captures real-time images or videos and identifies the user’s current emotional state.
- **Dashboard Display:** The web app presents a user-friendly dashboard where users can see their current and past emotions along with personalized advice.

The app aims to promote emotional awareness and provide guidance for mental well-being. Whether you’re looking to uplift your mood or enhance your mindfulness practice, this app supports you on your wellness journey.

# How to Open

**1. Ensure Connection to Mongo**

Download the MongoDB for VSC extension and add the database url: mongodb+srv://nsb8225:thefixers2.1@cluster0.oqt4t.mongodb.net/ when prompted to connect to the database.

**2. Create a new virtual environment following the commands:**

```
python3 -m venv .venv

```

**Mac** 
```
source .venv/bin/activate
```

**Windows**
```
.venv\Scripts\activate
```

**3. Install Dependencies if not already installed**

```
pip install opencv-python-headless
pip install requests
pip install pymongo
```

**4. Docker Compose**

Make sure that the Docker Desktop is downloaded and you are logged into your account before running the following comand:

```
docker-compose up --build

```

**5. Open the local host link for web-app and enjoy!**

To re-capture an emotion you will have to reload the page instead of pressing capture emotion again, bug to be fixed in later versions!

Notes for Usage: Face cannot be too far from camera (shoulders should be at bottom of screen) in order for the model to be able to read your emotion!

Thank you for the [Emotion Detection Model](https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection)!
