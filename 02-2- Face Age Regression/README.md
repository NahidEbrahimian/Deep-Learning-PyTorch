##  Face Age Regression

# Mnist Persian

- Mnist Persian classification using Pytorch


- [x] train.py
- [x] model.py
- [x] config.py
- [x] data_loader.py
- [x] mnist_persian.ipynb
- [ ] test.py (evaluate)
- [x] inference.py
- [x] requirements.txt

#

### Dataset

UTKFace dataset contain 9780 file

Dataset link: [utkface-new](https://www.kaggle.com/jangedoo/utkface-new)


### Instalation

1- Clone this repository using the following command:

`
https://github.com/NahidEbrahimian/Deep-Learning-Course-PyTorch
`

2- In `.Deep-Learning-Course-PyTorch/02-2- Face Age Regression` directory, run the following command to install requirements:

`
pip install -r requirements.txt
`
#

### Train

1- Clone this repository using the following command:

`
https://github.com/NahidEbrahimian/Deep-Learning-Course-PyTorch
`

2- Download dataset from this link: [utkface-new](https://www.kaggle.com/jangedoo/utkface-new) and put in `.Deep-Learning-Course-PyTorch/02-2- Face Age Regression` directory


3- Extract file using the following command:

`
!unzip -qq utkface-new.zip
`

4- Run training with the following command in `.Deep-Learning-Course-PyTorch/02-2- Face Age Regression` directory. you must select your device and set dataset directory

`
python3 train.py --device cuda --data_path ./crop_part1
`

#

### Test

- For evaluation, run the following command in `.Deep-Learning-Course-PyTorch/02-1- Mnist Persian` directory. you must select your device.

`
python3 test.py --device cuda --data_path ./MNIST_persian
`
#

### Inference

- For inference, you can run the following command in `.Deep-Learning-Course-PyTorch/02-2- Face Age Regression` directory.

input_img --> input image directory

`
python3 inference.py --device cuda --input_img 1.jpg
`

