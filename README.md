# Tech-Titans-AQUAPONICS-

# Aquaponics System with AI-Powered Decision Support
Overview
The envisaged project employs an AI approach for optimizing the management and performance of the aquaponics system with a decision-support system that is driven by real-time sensor data to predict and automate optimum control actions for maintaining desired water quality conditions. The overall task is split into two primary constituents.

1. **Standard LSTM Neural Network Implementation**: Long Short-Term Memory model development and training for action predictions based on time-series sensor readings.
2. **Intel-Optimized Version**: With access to optimized libraries from Intel, the training and inference of models will be improved, such as a TensorFlow and scikit-learn that provide better performance and scalability.

### **Objectives:**
• Develop a predictive model that will recommend appropriate actions on the basis of sensor data both past and current.
• Draw insights on improvement of performance using specialized libraries of Intel on deep learning as well as machine learning.

---

## **1. Dataset Overview**
This project is based on a **Sensor Based Aquaponics Fish Pond Dataset**, published on Kaggle. The dataset is rich in time-series data from the reading of sensors monitoring an aquaponics system. The parameters collected include:
 
- **Temperature**: Water temperature in Celsius.
- **pH**: The measure of acid or alkali in the water.
- **Dissolved Oxygen (DO)**: Concentration of oxygen in mg/L, thus reporting the availability of oxygen for the fish.
- **Ammonia**: Ammonia concentration in parts per million (ppm).
- **Flow Rate**: Rate at which water is flowing through the system.

### Data Characteristics:
The dataset is already in chronological order, which makes it well suited for sequence modeling with recurrent neural networks such as LSTMs. It also has a column manually labeled `action`, indicating required system interventions, such as nutrient adjustments, balancing water flow, or pH levels.
  
### Dataset Link:
The dataset can be accessed and downloaded from Kaggle</https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets>.
Purpose:
Data Preprocessing is an important process to transform raw data sensed from the sensors in a form that is easily usable for the models of machine learning. Data preprocessing includes:
 
1.  Missing Values Handling: Fill or interpolate missing entries from sensor readings that appear missing or inconsistent.
2.  Data Normalization: In such datasets, sensor readings have different units of measurement and scales, such as temperature and pH. Normalize values using techniques like Min-Max scaling.
3. **Feature Engineering**: Select only appropriate features such as `temperature`, `pH`, `DO`, `ammonia`, and `flow` and remove redundant columns.
4. **Encoding the Target Variable**: Encode categorical action names to numerical values such as 'increase_nutrient', 'turn_on_pump'
5. **Sequence Generation for LSTM**: LSTM needs sequences as input and returns an invariant time window. This includes generating sequences of sensor readings with the respective actions to train the model as well.

### Output of Preprocessing:
The final preprocessed data is comprised of time-based sequences of sensor readings together with an encoded target label for each sequence. This is fed into the LSTM model in this input format.

---

## **3. Building the LSTM Model**
### **Model Overview:**
The LSTM model is one kind of recurrent neural network (RNN) designed for capturing dependency in sequential data. It is suitable for time-series prediction and classification, in which historical data affects the future state. The following sections discuss applying an LSTM to learn sensor data patterns so as to predict the best action for the system.

### Model Architecture
The architecture of the LSTM model can be described below:
1. Input Layer: Provide sequences of timesteps, along with features like temperature, pH, and dissolved oxygen.
2. Hidden LSTM Layers: Two LSTM layers stacked to capture temporal dependencies.
3. Dropout Layers: Used to avoid overfitting and therefore enhance generalization
4. Dense Layers: Fully connected layers for transforming the LSTM outputs into a probability distribution over possible actions.
5. Output Layer: Employed softmax activation, multi-class classification so as to predict the appropriate action.

### Model Purpose:
The trained LSTM model will, in real-time, predict the control actions to be carried out given the sequence of sensor readings most recently registered.

---

## 4. Utilizing Intel-Optimized Libraries
### Objective:
For this, they have various specialized libraries, like **Intel TensorFlow** and **Intel Extension for Scikit-learn**, which are basically conceived to take a thrust in machine learning model performance on Intel hardware by optimizing deep learning operations, like matrix multiplications, data transformation using low-level optimizations.

### **Intel TensorFlow:**
Intel publishes a distribution of TensorFlow: `intel-tensorflow`. They aim at accelerating performance of deep learning models on CPUs. In the course of this project we are adding `intel-tensorflow` to speed up training and inference of the LSTM model.

### **Intel Extension for Scikit-Learn:**
This extension (`scikit-learn-intelex`) patches general operations that occur in scikit-learn, such as data pre-processing and model training, thus reducing the execution time. We can improve certain parts of the workflow through incorporating this library-for example, scaling data and encoding labels.

### **Performance Benefits:**
1. **Faster Model Training**: Optimizations by Intel result in reducing the times to train the model by up to an order of magnitude on Intel hardware.
2. **Fewer Resources Used**: Better use of the CPU means usage on lower-power devices in the real world.

---

## Model Training and Evaluation
### Training:
The trained LSTM model is fit to use preprocessed time series. The model learns to approximate the relationship for sequences of sensor readings and corresponding actions while training. Hyperparameters like the number of LSTM layers, units, dropout rates, and epochs are finely tuned for optimal outcomes.

### Evaluation:
Evaluation Metrics
Standard metrics of accuracy, precision, and recall are used with the trained model. The focus of evaluation is on the prediction capability of the model in predicting correct actions on unseen test sequences.

### **Intel-Optimized Training:**
The project documents and compares, through training time, memory usage, and CPU utilization against standard TensorFlow, using Intel's libraries.

---


## **6. Performance Comparison**
### **Objective:**
The performance of the standard LSTM model is compared against the variant optimized by Intel. Metrics for comparison are defined below:

1.  **Training Time**: Compare reduction in training time using TensorFlow that was optimized by Intel.
2.  **Inference Speed**: Determine how quickly the model can make predictions for new sensor sequences.
3.  **Resource Utilization**: Compare the CPU and memory usage during training and inference.

### Results
The results are presented in a table or graph so that the benefits from the optimizations by Intel can be visibly noticed.

---

## **7. Running the Project**
### **Requirements:**
 Install `intel-tensorflow`, `scikit-learn-intelex`, `pandas`, and any other required libraries.
 Download and preprocess your dataset.

### **Steps:**
1. **Download the Dataset**: Keep the dataset within the project directory.
2. **Preprocess the Data**: Run the script for data preprocessing.
3. **Training LSTM Model**: I train the model through the use of standard TensorFlow.
4. **Intel-TensorFlow**: Run the training script through Intel's optimized libraries.
5. **Performance Comparison**: Record the output.
 
--- 



  

## **8. Future Work**
- **Reinforcement Learning**: A reinforcement learning-based decision-making system shall be developed that can learn autonomously.
- **Real-Time Deployment**: The model shall then be integrated with live IoT sensors for real-time decision support.
- **IoT Automation**: These predictions can then automate the physical elements such as pumps and nutrient dispensers.

  ## 9. Block diagram and stimulation

  ![image](https://github.com/user-attachments/assets/f86ea48f-3855-40da-bfc6-d03cafba7cbc)
  
  Integrating a simulated pH sensor in the Proteus software can help visualize and validate the functioning of your aquaponics system before deploying physical components.
  This provides us the basic idea of how the senors works.

  ![image](https://github.com/user-attachments/assets/543b4d11-4f32-4561-861f-15264b3059c5)

  This is an general blockdiagram of how the system works:
  Raw Sensors:

This block represents various sensors placed in the aquaponics system to measure key environmental parameters like pH, temperature, dissolved oxygen (DO), and ammonia levels.
The sensors generate real-time data that will be used to monitor the health and balance of the aquatic environment.
ESP 8266:

The ESP 8266 is a Wi-Fi-enabled microcontroller that captures the raw sensor data and sends it to the cloud platform for further processing.
It acts as a bridge between the physical sensors and the data processing pipeline by transmitting sensor readings via the internet.
Adafruit IO:

Adafruit IO is a cloud platform used to collect, visualize, and store sensor data.
It receives data from the ESP 8266 and organizes it into structured datasets, making it accessible for machine learning algorithms.
Integrating Dataset:

This step involves aggregating the incoming data streams from different sensors into a single dataset.
It ensures that all sensor readings are synchronized and formatted properly for training the AI model.
Data Split:

The integrated dataset is split into training and testing datasets.
The training dataset is used to train the AI model, while the testing dataset is reserved for evaluating the model's performance.
Train Data:

This block represents the dataset prepared for training the model. The training data contains historical records of sensor readings and the corresponding actions taken.
Algorithm (LSTM):

The AI algorithm used here is an LSTM (Long Short-Term Memory) network, a type of recurrent neural network (RNN) suited for time-series analysis.
LSTMs are ideal for analyzing sequential data like sensor readings, capturing temporal dependencies to predict necessary actions.
Trained Model:

After the training process, the LSTM algorithm produces a trained model capable of making predictions based on new input data.
The model learns patterns in sensor data and identifies which actions (e.g., increasing nutrients, adjusting water flow) are needed to maintain a balanced system.
Test Data:

The testing dataset is fed into the trained model to validate its accuracy.
The test data contains unseen sequences of sensor readings that the model has not encountered before, helping to assess its generalization capability.
Prediction Results:

The trained model outputs predictions based on the test data, suggesting the optimal actions for given sensor conditions.
These predictions are compared against the expected outcomes to measure the model’s performance.
Calculate Model Performance:

This step involves calculating various performance metrics, such as accuracy, precision, and recall, to evaluate how well the model performs.
It helps determine if the model is reliable enough for real-world implementation.
Final Model:

Once the model is evaluated and tuned, it is finalized for deployment.
The final model is the one that will be used to make real-time decisions based on live sensor data.
Result (Alert if Required):

The final output of the system is an action or alert based on the sensor readings.
If the system detects any anomaly or condition that requires intervention (e.g., a sudden drop in pH), it can trigger an alert or automatically adjust system parameters.


  


## 10. References***
Sensor Based Aquaponics Fish Pond Dataset: [Kaggle Link](https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets).
Intel oneAPI Documentation: [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html).
