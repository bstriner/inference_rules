from inference_rules.tf.merge_walks import write_merged_walks


def main():
    output_path = 'output/FB15K/merged_walks/valid'
    walk_path = 'output/FB15K/walks/valid.pickle'
    splits = 1
    depth = 2
    write_merged_walks(output_path=output_path, walk_path=walk_path, splits=splits, depth=depth)


if __name__ == '__main__':
    main()
