def print_node_feature_slice_dataset_info(
    self,
    features,
    partition_features,
    num_features,
    overlap_features,
    num_unique_features,
) -> None:
    print("Number of clients = ", self.num_partitions)
    print("percentage overlap  = ", self.overlap_percent)

    print("number of total nodes = ", features.shape[0])
    print("Number of nodes in each client = ", partition_features.shape[0])

    print("Number of total features = ", num_features)

    print("Number amount of feature overlap = ", overlap_features[0].shape[0])
    print(
        "Number of split node features in each client = ",
        partition_features.shape[1],
    )
    print("Features lost in split = ", (num_features - (num_unique_features)))
