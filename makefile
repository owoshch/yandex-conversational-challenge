fasttext:
	git clone https://github.com/facebookresearch/fastText.git
	cd fastText && pip install .
	python -c "import fastText"
	python -c "from fastText import load_model"


corpus:
	wget -P ../data/ "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ru.zip"
	unzip ../data/wiki.ru.zip -d ../data/
	rm ../data/wiki.ru.zip


dataset:
	wget -P ../ "https://www.dropbox.com/s/ie9g8pkwyuul3sv/general-conversation-challenge-yandex-alice.zip"
	unzip ../general-conversation-challenge-yandex-alice.zip -d ../data/
	rm ../general-conversation-challenge-yandex-alice.zip


data:
	wget -P ../ "https://www.dropbox.com/s/6lgw599r2d967q3/preprocessed_data.zip"
	unzip ../preprocessed_data.zip -d ../data/
	rm ../preprocessed_data.zip

weights:
	wget -P ./ "https://www.dropbox.com/s/74x5gootj2fiypp/bilstm_regression_model.zip"
	unzip ./bilstm_regression_model.zip -d ./results/
	rm ./bilstm_regression_model.zip

run:
	python build_data.py
	python train_test_split.py
	python get_indices.py
	python separate_train_final.py