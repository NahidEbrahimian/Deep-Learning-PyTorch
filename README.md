# 01- Fashion Mnist Dataset

- [x] train.py
- [x] model.py
- [x] config.py
- [x] fashion_mnist.ipynb
- [x] test.py (evaluate)
- [x] inference.py
- [x] requirements.txt

#

### Train

- You can run training with the following command. you must select your device.

`
python3 train.py --device "cuda"
`
#

### Test

- For evaluation, run the following command. you must select your device.

`
python3 test.py --device "cuda"
`
### Inference

- For inference, you can run the following command.

input_img --> input image directory

`
python3 inference.py --device cuda --input_img 21.png
`
