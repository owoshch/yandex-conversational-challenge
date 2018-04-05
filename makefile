
fasttext:
    wget  ../data/ https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ru.vec


run:
    python build_data.py
    python get_indices.py
    python train.py