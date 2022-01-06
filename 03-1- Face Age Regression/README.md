##  Face Age Regression

- Face Age Regression using Pytorch - Transfer Learning


- [x] train.py
- [x] model.py
- [x] config.yaml.py
- [x] data_loader.py
- [x] transfer learning.ipynb
- [x] transfer learning-wandb_sweep.ipynb
- [x] test.py (evaluate)
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

2- In `.Deep-Learning-Course-PyTorch/03-1- Face Age Regression` directory, run the following command to install requirements:

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

4- Run training with the following command in `.Deep-Learning-Course-PyTorch/03-1- Face Age Regression` directory. you must select your device and set dataset directory

`
python3 train.py --device cuda --data_path ./crop_part1
`

#

### Test

1- Clone this repository using the following command:

`
https://github.com/NahidEbrahimian/Deep-Learning-Course-PyTorch
`

2- Download dataset from this link: [utkface-new](https://www.kaggle.com/jangedoo/utkface-new) and put in `.Deep-Learning-Course-PyTorch/02-2- Face Age Regression` directory


3- Extract file using the following command:

`
!unzip -qq utkface-new.zip
`

4- Run the following command in `.Deep-Learning-Course-PyTorch/03-1- Face Age Regression` directory. you must select your device.

`
python3 test.py --device cuda --data_path ./crop_part1
`
#

### Inference

- For inference, you can run the following command in `.Deep-Learning-Course-PyTorch/03-1- Face Age Regression` directory.

input_img --> input image directory

`
python3 inference.py --device cuda --input_img 1.jpg
`

