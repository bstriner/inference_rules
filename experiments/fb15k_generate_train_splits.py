import numpy as np

from inference_rules.generate_splits import generate_splits


def main():
    dataset_path = 'output/FB15K/dataset'
    output_path = 'output/FB15K/splits'
    data = np.load('{}/dataset.npz'.format(dataset_path))
    train = data['train']
    splits = 20
    generate_splits(output_path, train=train, splits=splits)


if __name__ == '__main__':
    main()
