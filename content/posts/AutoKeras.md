---
title: "AutoKeras - A Quick Introduction and Demostration"
date: 2024-10-31T13:55:03-04:00
draft: false # Set 'false' to publish
tableOfContents: true # Enable/disable Table of Contents
description: 'This post shows you how to quickly install and build a demo with AutoKeras'
categories:
  - Tutorials
tags:
  - AutoKeras
  - TensorFlow
---

## Getting started with AutoKeras
AutoKeras is an open-source AutoML library built on top of TensorFlow, designed to simplify the process of building and training deep learning models without requiring extensive knowledge of model architecture and hyperparameter tuning. It is particularly helpful for tasks like image classification, text classification, and more, offering a user-friendly approach to machine learning.

AutoKeras streamlines several challenges in machine learning:

- **Model Selection and Hyperparameter Tuning**: AutoKeras automates the process of searching for the best model architecture and hyperparameters, making it suitable for those who may not have deep expertise in deep learning.
- **Ease of Use**: The library offers a high-level API that reduces complexity, allowing users to get up and running quickly.
- **End-to-End ML Pipelines**: AutoKeras provides tools to build, train, and deploy models seamlessly.

### What problem does AutoKeras address
AutoKeras addresses several key problems in machine learning:

- **Model Selection and Hyperparameter Tuning**: Selecting the right model architecture and tuning hyperparameters can be complex and time-consuming. AutoKeras automates these processes, making it easier for non-experts to develop effective models without extensive experimentation.
- **Accessibility for Beginners**: Building deep learning models traditionally requires substantial knowledge of neural network architectures and optimization strategies. AutoKeras provides a high-level, user-friendly interface that simplifies the process, making machine learning more accessible to beginners.
- **Efficiency and Speed**: Searching for the best model architecture and hyperparameters manually can be inefficient. AutoKeras automates this search, significantly reducing the time and computational resources needed to develop well-performing models.
- **Reducing Development Effort**: By automating model design and optimization, AutoKeras reduces the effort required to develop machine learning models, allowing researchers and developers to focus more on data analysis and the specific needs of their applications.
- **End-to-End Machine Learning**: AutoKeras provides tools to handle the entire machine learning pipeline, from data preprocessing to model training and evaluation, making the workflow seamless and efficient.

Overall, AutoKeras simplifies and accelerates the process of developing and deploying deep learning models, making it easier for users to leverage the power of machine learning in various applications.
### Strength and Limitations

Here are the strengths and limitations of AutoKeras:

#### **Strengths of AutoKeras:**

1. **Ease of Use**: AutoKeras provides a high-level API that abstracts away the complexities of deep learning model building and hyperparameter tuning. Users can quickly create models without needing extensive expertise in machine learning.

2. **Automated Model Search**: The automated search for the best model architecture and hyperparameters significantly reduces the effort and time required for model selection. This feature is particularly useful for beginners or those unfamiliar with designing deep learning architectures.

3. **End-to-End Automation**: AutoKeras handles the entire machine learning workflow, from data preprocessing to model training and evaluation. This streamlining is helpful for prototyping and rapid experimentation.

4. **Support for Various ML Tasks**: AutoKeras supports multiple machine learning tasks, such as image classification, text classification, regression, and more, making it versatile for different use cases.

5. **Integration with TensorFlow and Keras**: As an extension of the TensorFlow and Keras ecosystem, AutoKeras benefits from TensorFlow's extensive tools for deployment, scalability, and production-readiness.

6. **Hyperparameter Optimization**: AutoKeras includes built-in hyperparameter optimization, which can improve model performance by finding the best settings for training without manual intervention.


#### **Limitations of AutoKeras:**

1. **Limited Customization**: Although AutoKeras simplifies the model-building process, it comes at the cost of reduced control over model architecture and hyperparameter choices. Advanced users may find it challenging to make highly specific customizations for complex use cases.

2. **Resource-Intensive**: The automated search for the best model and hyperparameters can be computationally expensive, requiring significant resources, especially for large datasets or complex tasks.

3. **Potentially Longer Training Time**: Because AutoKeras explores different model architectures and hyperparameters, it may take longer to train compared to a manually designed model that uses known good configurations.

4. **Scalability Challenges**: While AutoKeras is excellent for prototyping and small- to medium-scale projects, it may not be the best choice for very large datasets or high-scale deployments where fine-tuned control and optimization are needed.

5. **Less Optimal for Complex Problems**: For highly specialized or complex problems, manually designed models may still outperform those generated by AutoKeras. The automated approach may not always yield the best possible model.

6. **Dependency on TensorFlow**: AutoKeras relies on TensorFlow as its backend, so users who prefer or are required to use other frameworks (such as PyTorch) may find it limiting.

7. **Limited Interpretability**: Understanding the logic behind the model architecture decisions made by AutoKeras can be challenging, which may affect model interpretability and debugging.

## Demo
In this tutorial, we’ll demonstrate how to use AutoKeras to build a text classification model to predict movie genres based on their overviews. We will walk through data preprocessing, model training, and evaluation.
You can find the source code [here](https://github.com/s1monFu/autokera_demo)
### Requirements

- A machine that supports TensorFlow. e.g. Mac has limited support over TensorFlow and AutoKeras.
- Python 3: Choose Python version between 3.8-3.11.
- Pip
- Tensorflow >= 2.3.0: AutoKeras is based on TensorFlow. Please follow this [tutorial](https://www.tensorflow.org/install) to install TensorFlow for python3.
- pandas
- numpy
- scikit-learn
- GPU Setup (Optional): If you have GPUs on your machine and want to use them to accelerate the training, you can follow this tutorial to setup.
- Conda or other environment management system ready (Optional)

### Set up
(Optional) Create a new environment using Conda or a similar tool
```
conda create --name ak python==3.11
conda activate ak
```

Then install necessary packages
```
pip install --upgrade pip
pip install tensorflow
pip install git+https://github.com/keras-team/keras-tuner.git
pip install autokeras
pip install pandas
pip install numpy
pip install scikit-learn
```

### Data Preparation

We will use a sample movie dataset that contains movie overviews and their corresponding genres. Our goal is to classify movies based on their overviews.
The movie dataset can be found [here](https://github.com/s1monFu/autokera_demo/blob/main/movies.csv)
Here’s the code for loading and preparing the data:

```python
import pandas as pd
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import tensorflow as tf
import keras.backend as K

# Clear previous sessions to free up resources
K.clear_session()

# Step 1: Read the dataset
df = pd.read_csv('./movies.csv')

# Step 2: Process the genres
df['genres_list'] = df['genres'].str.split('|')

# Extract the primary genre
df['primary_genre'] = df['genres_list'].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
)

# Remove rows with missing primary_genre or overview
df = df[df['overview'].notnull() & df['primary_genre'].notnull()]

# Ensure that overview is a string
df['overview'] = df['overview'].astype(str)
```

### Filtering and Preprocessing

We need to filter and preprocess the data to ensure a balanced distribution of genres:

```python
# Step 3: Prepare features and labels
x = df['overview'].values
y = df['primary_genre'].values

# Combine x and y into a DataFrame for easier manipulation
data = pd.DataFrame({'overview': x, 'primary_genre': y})

# Limit the number of samples for this demonstration
data = data.iloc[:100]

# Filter genres with at least 2 samples
class_counts = data['primary_genre'].value_counts()
classes_with_enough_samples = class_counts[class_counts >= 2].index.tolist()
data = data[data['primary_genre'].isin(classes_with_enough_samples)]

# Shuffle the data for randomness
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract x and y from the DataFrame
x = data['overview'].values
y = data['primary_genre'].values

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Verify class counts after filtering
class_counts_after = Counter(y_encoded)
print("Class counts after filtering:", class_counts_after)

# Convert to proper types
x = np.array(x, dtype=str)
y_encoded = np.array(y_encoded, dtype=int)
```

### Splitting the Data

We split the data into training and testing sets:

```python
# Step 4: Split into training and testing sets with stratification
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
```

---

## Building the Model with AutoKeras

Now, let's define and train a text classification model using AutoKeras:

```python
# Step 5: Initialize the text classifier
clf = ak.TextClassifier(overwrite=True, max_trials=1)

# Train the classifier
clf.fit(
    x_train,
    y_train,
    validation_split=0.15,
    epochs=1,
    batch_size=2,
)
```

### Evaluating the Model

Once the model is trained, we can evaluate its performance:

```python
# Step 6: Evaluate the classifier
predicted_y = clf.predict(x_test)
evaluation = clf.evaluate(x_test, y_test)
print(f"Evaluation Results: {evaluation}")

# Optional: Decode the predicted labels back to genres
predicted_genres = le.inverse_transform(predicted_y.flatten().astype(int))
print(f"Predicted Genres: {predicted_genres}")
```

---

## Conclusion

AutoKeras offers an efficient and user-friendly way to build and train machine learning models. In this tutorial, we demonstrated how to use AutoKeras for a text classification problem, automating the model selection and tuning processes. AutoKeras simplifies the model-building experience, making it accessible even to those new to deep learning. While it provides convenience and speed, always consider potential trade-offs, such as limited fine-tuning options for highly customized models.

By leveraging AutoKeras, you can focus more on understanding your data and less on the intricacies of deep learning model design.



