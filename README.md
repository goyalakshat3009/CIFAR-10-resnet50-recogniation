# CIFAR-10 Object Recognition using ResNet50

An image classifier that recognises 10 everyday object classes, built with **transfer learning** on an ImageNet-pretrained **ResNet50** and deployed as a **Streamlit** web app.

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Live demo:** _add your Streamlit app link here_

## Results

| Metric | Value |
|---|---|
| Test accuracy | **93.76%** (5,000 held-out images) |
| Best validation accuracy | 93.64% |
| Weakest classes | cat (F1 0.88), dog (F1 0.89) |
| Strongest classes | ship, frog (F1 0.97) |

## How it works

1. **Data:** the 50,000 labelled CIFAR-10 images from the [Kaggle CIFAR-10 competition](https://www.kaggle.com/c/cifar-10), split 80 / 10 / 10 into train (40,000), validation (5,000) and test (5,000), stratified by class.
2. **Preprocessing:** ResNet50's own `preprocess_input`. Images are upscaled from 32x32 to 96x96 inside the model.
3. **Model:** pretrained ResNet50 (no top) + global average pooling + dropout (0.3) + a 10-class softmax layer. A random horizontal flip is used for augmentation.
4. **Training in two phases:**
   - Phase 1: train only the new classifier head with the backbone frozen (5 epochs, learning rate 1e-3).
   - Phase 2: unfreeze the backbone, keeping BatchNorm layers frozen, and fine-tune (5 epochs, learning rate 1e-5).
5. **Deployment:** the saved model is loaded by a Streamlit app that accepts an uploaded or camera photo and shows the predicted class with its confidence and the top 3 probabilities.

## Project structure

```
.
├── app.py                                   # Streamlit web app
├── requirements.txt                         # Python dependencies for deployment
├── cifar10_resnet50.keras                   # Trained model (stored with Git LFS)
├── CIFAR-10 ResNet50 Kaggle Easy.ipynb      # Training notebook
├── samples/                                 # Optional example photos for the app
├── .gitattributes                           # Git LFS tracking rules
└── .gitignore
```

## Run locally

```bash
git clone https://github.com/goyalakshat3009/CIFAR-10-resnet50-recogniation.git
cd CIFAR-10-resnet50-recogniation
pip install -r requirements.txt
streamlit run app.py
```

The model file is stored with [Git LFS](https://git-lfs.com), so install it and run `git lfs install` before cloning.

## Retrain the model

1. Create a Kaggle API token (Kaggle > Settings > API > Create New Token) and save `kaggle.json` in your Downloads folder.
2. Accept the rules of the [CIFAR-10 competition](https://www.kaggle.com/c/cifar-10) once.
3. Open `CIFAR-10 ResNet50 Kaggle Easy.ipynb` and run all cells. It downloads and extracts the data, trains the model and saves `cifar10_resnet50.keras`.

A GPU is strongly recommended. TensorFlow has no native Windows GPU support from version 2.11, so use Google Colab, Kaggle notebooks or WSL2. On CPU, training took about 3.5 hours.

## Limitations

- The model only knows the 10 CIFAR-10 classes, so a photo of anything else is forced into one of them.
- Images are shrunk to 32x32 like the training data, so small, centred objects on simple backgrounds work best.
- Cats and dogs are the most often confused pair.

## Tech stack

Python, TensorFlow / Keras, NumPy, pandas, scikit-learn, Matplotlib, Streamlit

## Acknowledgements

- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky, via Kaggle
- Model: ResNet50 (He et al., 2015), ImageNet weights from Keras Applications
