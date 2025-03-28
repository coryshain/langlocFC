import os

data_path = input('Enter path to data repository (available at https://openneuro.org/datasets/ds006071) >>> ')
with open('data_path.txt', 'w') as f:
    f.write(data_path)