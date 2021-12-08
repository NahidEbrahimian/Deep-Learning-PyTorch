# 01- Fashion Mnist Dataset

- Fashion Mnist dataset classification using Pytorch

- [x] train.py
- [x] model.py
- [x] config.py
- [x] fashion_mnist.ipynb
- [x] test.py (evaluate)
- [x] inference.py
- [x] requirements.txt

#

### Instalation

1- Clone this repository using the following command:

`
https://github.com/NahidEbrahimian/Deep-Learning-Course-2
`

2- In `.Deep-Learning-Course-2/01- Fashion Mnist` directory, run the following command to install requirements:

`
pip install -r requirements.txt
`
#

### Train

- You can run training with the following command in `.Deep-Learning-Course-2/01- Fashion Mnist` directory. you must select your device.

`
python3 train.py --device cuda
`
#

### Test

- For evaluation, run the following command in `.Deep-Learning-Course-2/01- Fashion Mnist` directory. you must select your device.

`
python3 test.py --device cuda
`
#

### Inference

- For inference, you can run the following command in `.Deep-Learning-Course-2/01- Fashion Mnist` directory.

input_img --> input image directory

`
python3 inference.py --device cuda --input_img 21.png
`
