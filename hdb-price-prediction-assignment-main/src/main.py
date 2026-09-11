from helper.data_splitter import DataSplitter

def main():
    # step 1) split dataset into train, validate + model selection, final testing
    data_splitter = DataSplitter()
    train, validate, test = data_splitter.split()
    print(f"{len(train)=}")
    print(f"{len(validate)=}")
    print(f"{len(test)=}")
    return

main()