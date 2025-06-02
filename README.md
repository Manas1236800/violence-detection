# violence-detection
This deep learning model is built for real-time processing and detect violence using LSTM and CNN for frame processing and realtime processing.
this model uses a violence detection dataset from kaggle :
 https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

This repository contains the code for a deep learning model designed to perform violence detection from short video sequences. The model leverages the power of pre-trained convolutional neural networks (CNNs) for spatial feature extraction and Recurrent Neural Networks (RNNs) (specifically LSTMs) for temporal sequence modeling.

## 🚀 Project Overview

The goal of this project is to classify human activities given a sequence of video frames. This is a common task in computer vision with applications in surveillance, sports analysis, human-computer interaction, and more.

## ✨ Model Architecture & Techniques

The model is built using Keras/TensorFlow and incorporates several key deep learning techniques:

* **Hybrid CNN-RNN Architecture**: Combines the strengths of CNNs for extracting spatial features from individual frames and RNNs for learning dependencies across time in the sequence of frames.
* **Transfer Learning with MobileNetV2**:
    * A pre-trained `MobileNetV2` model, initialized with `ImageNet` weights, is used as the base for feature extraction. This significantly speeds up training and improves performance on limited datasets by leveraging knowledge learned from a vast image dataset.
    * The `include_top=False` argument ensures that the classification head of `MobileNetV2` is removed, allowing us to add our custom classification layers.
    * The `input_shape=(128, 128, 3)` specifies the expected input size for individual frames.
    * **Preventing Initial Downsampling**: The `base_model.layers[0].strides = (1, 1)` line is a crucial optimization to prevent downsampling in the very first convolutional layer of MobileNetV2. This helps preserve more spatial information from the input frames, which can be beneficial for activities that involve fine-grained movements.
    * **Frozen Base Model Layers**: The layers of `MobileNetV2` are set to `trainable = False`. This means their pre-trained weights will not be updated during training, effectively using MobileNetV2 as a fixed feature extractor. This prevents catastrophic forgetting and allows the model to quickly adapt to the new task.
* **`TimeDistributed` Wrapper**:
    * The `TimeDistributed` layer is used to apply the `MobileNetV2` base model and the `GlobalAveragePooling2D` layer independently to each frame in the input sequence. This is essential for processing video data where each frame needs to be processed by a CNN.
    * Input to `TimeDistributed(base_model)` is `(batch_size, 15, 128, 128, 3)` (15 frames, each 128x128x3).
    * Output of `TimeDistributed(GlobalAveragePooling2D)` converts the spatial features from each frame into a fixed-size vector, effectively flattening the (height, width, channels) dimensions into a single vector of 1280 features (for MobileNetV2).
* **LSTM Layers for Temporal Modeling**:
    * Two `LSTM` (Long Short-Term Memory) layers are used to process the sequence of features extracted from each frame. LSTMs are well-suited for capturing long-range dependencies in sequential data.
    * The first LSTM layer (`LSTM(128, return_sequences=True)`) processes the sequence and outputs a sequence of states, passing contextual information to the next LSTM layer.
    * The second LSTM layer (`LSTM(64, return_sequences=False)`) processes the output of the first LSTM and returns only the last output, summarizing the entire sequence's temporal information.
* **Dense (Fully Connected) Layers**:
    * Several `Dense` layers are stacked after the LSTMs to learn complex non-linear relationships from the temporal features.
    * Activations are `relu` for hidden layers.
* **Batch Normalization**:
    * `BatchNormalization` layers are strategically placed after `Dense` layers. This technique normalizes the activations of the previous layer, reducing internal covariate shift, stabilizing the learning process, and often leading to faster convergence and better performance.
* **Dropout**:
    * `Dropout` layers are used to prevent overfitting by randomly setting a fraction of input units to zero at each update during training. This forces the network to learn more robust features.
* **Binary Classification Output**:
    * The final `Dense` layer has `1` unit and a `sigmoid` activation function, making it suitable for binary classification tasks (e.g., activity vs. no activity, or one specific activity vs. all others).
* **Optimizer and Loss Function**:
    * The model is compiled with the `adam` optimizer, a popular adaptive learning rate optimization algorithm.
    * `binary_crossentropy` is used as the loss function, which is standard for binary classification problems.
    * `accuracy` is used as the evaluation metric.

## ⚙️ How to Use (Code Snippet Explained)

The provided code snippet defines, builds, and compiles the model. To use this model:

1.  **Install Dependencies**: Make sure you have TensorFlow/Keras installed.
    ```bash
    pip install tensorflow keras
    ```
2.  **Prepare Your Data**:
    * Your input data should be in the shape `(batch_size, num_frames, height, width, channels)`.
    * For this model, `num_frames=15`, `height=128`, `width=128`, and `channels=3` (for RGB images).
    * Your target labels for binary classification should be `0` or `1`.
3.  **Integrate the Model**
## 📈 Next Steps

* **Data Preparation**: Load and preprocess your video datasets into the required `(num_frames, height, width, channels)` format.
* **Training**: Train the model on your prepared dataset.
* **Evaluation**: Evaluate the model's performance on a separate test set.
* **Hyperparameter Tuning**: Experiment with different LSTM units, dense layer sizes, dropout rates, and learning rates.
* **Fine-tuning**: Consider unfreezing some layers of the `MobileNetV2` base model and fine-tuning them with a very low learning rate for potentially better performance, especially if your dataset is large and diverse.

---

**Author**: Manas Moolchandani
