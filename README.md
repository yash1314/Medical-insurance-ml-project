# Medical Insurance Price Predictor

- Predict medical insurance prices based on age, gender, children, BMI, region, and smoking status. This project uses Gradient boost Regressor machine learning model to provide estimated insurance costs.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Project Overview

The **Medical Insurance Price Predictor** is a machine learning project developed in Visual Studio Code. It predicts medical insurance costs for individuals based on several input features, including age, gender, number of children, BMI, region, and smoking status. This tool can be valuable for insurance companies and individuals to estimate insurance costs.

## Folder Structure

The project follows a well-organized folder structure:

- `artifacts/`: Stores generated artifacts and model checkpoints.
- `notebooks/`: Contains Jupyter notebooks for data analysis and experimentation.
- `logs/`: Logs related to the project at every step.
- `src/`: Python source code files for the project.
  - `logger.py`: Logging functionality for the project.
  - `exception.py`: Custom exception classes for error handling.
  - `utils.py`: Utility functions used across the project.
  - `components/`: Modular components for data ingestion, transformation, and model training.
  - `pipelines/`: End-to-end pipelines for prediction and training.
- `images/`: Images or diagrams related to the project.
- `requirements.txt`: Lists project dependencies.
- `setup.py`: Setup script for the project.
- `app.py`: Main application script.
- `Dockerfile`: Docker container configuration.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://https://github.com/yash1314/industry-grade-ml-project.git
   cd medical-insurance-predictor

2. Install project dependencies:

pip install -r requirements.txt

## Usage
- Data Ingestion: Use the data ingestion component to load your data.
- Data Transformation: Transform and preprocess the data for training.
- Model Training: Train the machine learning model using the training pipeline.
- Prediction: Make predictions using the prediction pipeline.
- Dockerize the project for easy deployment and scalability.
- Run the project on an AWS EC2 instance via Streamlit for web-based interaction.
Detailed usage instructions are available in the project's notebooks and code documentation.

## Contributing
We welcome contributions! If you would like to contribute to this project, please review our contributing guidelines.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Thanks to the contributors of open-source libraries and frameworks used in this project. 

## Contact
If you have questions or feedback, feel free to reach out to:

Name: Yash keshari
Email: yashkeshari79@gmail.com 
