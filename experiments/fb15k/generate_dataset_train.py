from inference_rules.dataset import DatasetProcessor


def main():
    negative_samples = 50
    proc = DatasetProcessor('output/dataset', mode='train', negative_samples=negative_samples)
    proc.generate_dataset('output/processed/train.tfrecords')


if __name__ == '__main__':
    main()
