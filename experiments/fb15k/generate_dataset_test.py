from inference_rules.dataset import DatasetProcessor


def main():
    negative_samples = 200
    proc = DatasetProcessor('output/dataset', mode='test', negative_samples=negative_samples)
    proc.generate_dataset('output/processed/test.tfrecords')


if __name__ == '__main__':
    main()
