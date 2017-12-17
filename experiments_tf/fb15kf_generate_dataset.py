from inference_rules.generate_dataset import generate_dataset


def main():
    input_path = 'output/filtered/{}.txt'
    dataset_path = 'output/FB15Kf/dataset'
    generate_dataset(input_path=input_path, output_path=dataset_path, map=[0, 2, 1])


if __name__ == '__main__':
    main()
