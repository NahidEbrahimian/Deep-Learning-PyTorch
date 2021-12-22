# Mnist Persian

- Mnist Persian classification using Pytorch


- [x] train.py
- [x] model.py
- [x] config.py
- [x] mnist_persian.ipynb
- [x] test.py (evaluate)
- [x] inference.py
- [x] requirements.txt

#

### Dataset

Dataset contain persian handwritten numbers of 0 to 9

dataset link: [persian_mnist](https://drive.google.com/drive/folders/14aDOVDrczXi1uRDb8FMbisJPdiGxNF_2?usp=sharing)

### Instalation

1- Clone this repository using the following command:

`
https://github.com/NahidEbrahimian/Deep-Learning-Course-PyTorch
`

2- In `.Deep-Learning-Course-PyTorch/02-1- Mnist Persian` directory, run the following command to install requirements:

`
pip install -r requirements.txt
`
#

### Train

- You can run training with the following command in `.Deep-Learning-Course-PyTorch/02-1- Mnist Persian` directory. you must select your device and set dataset directory

`
python3 train.py --device cuda --data_path ./MNIST_persian
`
#

### Test

- For evaluation, run the following command in `.Deep-Learning-Course-PyTorch/02-1- Mnist Persian` directory. you must select your device.

`
python3 test.py --device cuda --data_path ./MNIST_persian
`
#

### Inference

- For inference, you can run the following command in `.Deep-Learning-Course-PyTorch/02-1- Mnist Persian` directory.

input_img --> input image directory

`
python3 inference.py --device cuda --input_img 21.png
`

