Breast Cancer Classification using Neural Network
=================================================

Overview
--------

This project implements a Neural Network using TensorFlow and Keras to classify breast cancer tumors as **malignant** or **benign**. The model is trained on the **Breast Cancer Wisconsin dataset** from Scikit-learn.

Dataset
-------

The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**, which contains **30 numerical features** extracted from digitized images of fine needle aspirate (FNA) of breast masses. The target variable indicates whether the tumor is **malignant (0)** or **benign (1)**.

Installation
------------

To run this project, ensure you have the following dependencies installed:

    pip install numpy pandas matplotlib scikit-learn tensorflow keras

Project Workflow
----------------

1.  **Load Dataset:** Import the dataset from Scikit-learn.
    
2.  **Preprocess Data:** Convert to a Pandas DataFrame, standardize features using `StandardScaler`.
    
3.  **Split Dataset:** Divide data into training and test sets.
    
4.  **Build Neural Network:**
    
    *   Input Layer: 30 neurons (features from dataset)
        
    *   Hidden Layer: 20 neurons with ReLU activation
        
    *   Output Layer: 2 neurons with softmax activation
        
5.  **Compile Model:**
    
    *   Optimizer: Adam
        
    *   Loss Function: Sparse Categorical Crossentropy
        
    *   Metric: Accuracy
        
6.  **Train Model:** Train with a validation split of 10% and 10 epochs.
    
7.  **Evaluate Performance:** Plot accuracy and loss graphs.
    
8.  **Make Predictions:** Convert probabilities to class labels for new input data.
    

Model Training
--------------

The model is trained using the following command:

    history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

Results
-------

After training, the model is evaluated on the test data:

    loss, accuracy = model.evaluate(X_test_std, Y_test)
    print(f'Test Accuracy: {accuracy:.4f}')

Sample Prediction
-----------------

To predict the classification of a new tumor:

    input_data = np.asarray([...]) # Replace with feature values
    input_data_reshaped = input_data.reshape(1, -1)
    input_data_std = scaler.transform(input_data_reshaped)
    prediction = model.predict(input_data_std)
    
    prediction_label = np.argmax(prediction)
    if prediction_label == 0:
        print('The cancer is MALIGNANT')
    else:
        print('The cancer is BENIGN')

Performance Metrics
-------------------

*   Accuracy: **~95%** on the test dataset
*   ![image](https://github.com/user-attachments/assets/dee81353-8fed-4ebe-b30f-aa4c3663fbe4)

    
*   Loss: **Monitored using validation loss curve**
    ![image](https://github.com/user-attachments/assets/888499c5-99a2-4cb7-93d4-35935442c7bb)


Future Improvements
-------------------

*   Optimize hyperparameters (number of layers, neurons, activation functions)
    
*   Use **binary cross-entropy** with a single neuron instead of softmax
    
*   Implement early stopping to prevent overfitting
    

Author
------

\[Your Name\]

License
-------

This project is licensed under the MIT License.
===============================================
