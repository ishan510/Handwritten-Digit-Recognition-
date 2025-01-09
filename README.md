# **Handwritten Digit Recognition using Convolutional Neural Networks (CNN)**

## **Project Overview**
This project uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits from the **MNIST dataset** with a classification accuracy of **98.96%** on the test set. It demonstrates an end-to-end machine learning workflow, from data preprocessing to model deployment, showcasing the effectiveness of deep learning in image classification tasks.

---

## **Features**
- Preprocessed grayscale images of handwritten digits for input to the CNN.
- Designed and trained a deep learning model achieving high classification accuracy.
- Visualized training performance with metrics like accuracy and loss over epochs.
- Generated predictions for unseen data and prepared Kaggle submissions.
- Provided insights into misclassified digits for performance analysis.

---

## **Dataset**
- **Source**: [Kaggle - Digit Recognizer Dataset](https://www.kaggle.com/c/digit-recognizer)
- **Description**:
  - The dataset contains 70,000 grayscale images of handwritten digits (28x28 pixels).
  - Training set: 60,000 labeled images.
  - Test set: 10,000 unlabeled images.

---

## **Tech Stack**
- **Languages**: Python
- **Libraries**: TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, Kaggle

---

## **Project Workflow**
1. **Data Preprocessing**:
   - Reshaped input images to match CNN requirements (28x28x1).
   - Normalized pixel values to the range [0, 1].
   - Split training data into training and validation sets.
2. **Model Architecture**:
   - Embedding Layer:
     - Trainable embeddings and pretrained embeddings (GloVe).
   - Convolutional Layers:
     - Extracted local n-gram features.
   - Max-Pooling Layer:
     - Reduced dimensionality and focused on key features.
   - Dense Layers:
     - Performed classification into three sentiment classes.
3. **Evaluation**:
   - Achieved 98.96% accuracy on the test set.
   - Confusion matrix and F1-score to assess performance.
4. **Submission**:
   - Prepared predictions in the required Kaggle format for leaderboard evaluation.

---
