
# Horse or Human Image Classifier

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow to classify images as either horses or humans. The model is trained on a dataset of images and achieves significant accuracy after 15 epochs.

## Dataset

The dataset used consists of 500 horse images and 527 human images, downloaded from [Google Cloud Storage](https://storage.googleapis.com/learning-datasets/horse-or-human.zip).

## Project Structure

  - `horse-or-human.zip`: The dataset containing the images of horses and humans.
  - `train_horse_dir`: Directory containing the horse images for training.
  - `train_human_dir`: Directory containing the human images for training.
  - `model.py`: Script to define, compile, and train the model.

## Model Architecture

The model consists of the following layers:

   1. **Conv2D + MaxPooling2D**: 16 filters with 3x3 kernel, followed by a MaxPooling layer.
   2. **Conv2D + MaxPooling2D**: 32 filters with 3x3 kernel, followed by a MaxPooling layer.
   3. **Conv2D + MaxPooling2D**: 64 filters with 3x3 kernel, followed by a MaxPooling layer.
   4. **Conv2D + MaxPooling2D**: 64 filters with 3x3 kernel, followed by a MaxPooling layer.
   5. **Conv2D + MaxPooling2D**: 64 filters with 3x3 kernel, followed by a MaxPooling layer.
   6. **Flatten**: Flatten the 3D tensor to a 1D tensor.
   7. **Dense**: Fully connected layer with 512 neurons.
   8. **Dense**: Output layer with 1 neuron and a sigmoid activation function.

## Training

The model was compiled using the RMSprop optimizer and trained using binary_crossentropy as the loss function. It was trained for 15 epochs with a batch size of 128.

Sample accuracy results from training:

    Epoch 1: Accuracy = 0.54
    Epoch 15: Accuracy = 0.89

## Testing

After training, the model can be used to predict whether a new image is a horse or a human. Images are resized to 300x300 before prediction.
Usage

## Download and unzip the dataset:

```bash
!wget https://storage.googleapis.com/learning-datasets/horse-or-human.zip -O /tmp/horse-or-human.zip
```

## Extract the dataset:


    import zipfile
    with zipfile.ZipFile('/tmp/horse-or-human.zip', 'r') as zip_ref:
        zip_ref.extractall('/tmp/horse-or-human')

## Train the model using the following script:


     model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1)


## Test the model with new images by uploading them and running the prediction:


    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = model.predict(images, batch_size=10)

## Results

The model is capable of distinguishing between horses and humans with a final accuracy of 89%.


## Requirements

  - Python 3.x
  - TensorFlow
  -  NumPy
  -  Matplotlib

    pip install tensorflow numpy matplotlib


## Acknowledgments

The dataset used for training is provided by Google and can be accessed [here](https://storage.googleapis.com/learning-datasets/horse-or-human.zip).




