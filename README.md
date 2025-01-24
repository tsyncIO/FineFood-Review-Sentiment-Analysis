# Sentiment Analysis for Amazon Reviews

## Project Overview

This project leverages state-of-the-art natural language processing (NLP) techniques to analyze customer reviews from Amazon Fine Food Reviews. Using BERT and RoBERTa, the model extracts sentiment insights to understand customer satisfaction, identify trends, and make data-driven decisions. The system processes large-scale datasets in real-time and is deployed using AWS Lambda, GCP, BigQuery, and Spark for scalable and efficient processing.

### Key Features:

- **Real-time Sentiment Analysis** of customer reviews
- **Sentiment Classification**: Positive, Negative, or Neutral
- **Data Processing**: Large-scale, real-time processing with **Apache Spark** and **AWS Lambda**
- **Data Storage**: Store and manage datasets using **BigQuery** and **AWS RDS (PostgreSQL)**.
- **Model Deployment**: Deployed on **AWS Lambda** for scalable inference.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

To run this project locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```


### 2. Install dependencies
Create a virtual environment and install the required Python packages:

```
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt

```

### 3. Set up environment variables
Make sure to create a .env file in the root directory with the following variables:

```
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
GCP_PROJECT_ID=your-gcp-project-id

```


### 4. Usage
## 1. Running the Sentiment Analysis
The project includes a predict.py script that performs sentiment analysis on customer reviews.

```python predict.py --input "I love this product! It's fantastic." ```

This command will classify the sentiment of the provided review as Positive or Negative.

## 2. Real-time Data Processing with AWS Lambda
You can deploy the model to AWS Lambda for real-time inference. Use the deploy_lambda.py script to package and deploy the model.

```python deploy_lambda.py```

This script will upload your model and the required dependencies to AWS Lambda, allowing you to trigger sentiment analysis from API calls.

## 3. Scaling with Spark
Use Apache Spark for large-scale data processing. You can run the following command to process a large dataset:


```python process_data_with_spark.py --input "data/reviews.csv" --output "data/processed_reviews.csv"```




### Technologies Used
## BERT & ## RoBERTa for sentiment classification
## AWS Lambda for real-time inference
## GCP (Google Cloud Platform) for data storage and processing
## BigQuery for managing large-scale datasets
## Spark for distributed data processing
## PostgreSQL (AWS RDS) for storing processed data
## Python & ## Hugging Face Transformers for model implementation
## Flask (if API integration is required)

### Contributing
Feel free to fork this repository, submit issues, or send pull requests. Contributions are welcome!

1. Fork the project
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Create a new Pull Request

### License
This project is licensed under the MIT License - see the LICENSE file for details.






