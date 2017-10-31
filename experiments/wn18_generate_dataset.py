from inference_rules.generate_dataset import generate_dataset


def main():
    input_path = '../../data/kbcdata/WN18/{}.txt'
    output_path = 'output/dataset-WN18'
    map = [0, 2, 1]
    generate_dataset(input_path=input_path, output_path=output_path, map=map)


if __name__ == '__main__':
    main()
