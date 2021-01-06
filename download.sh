wget https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_morph_news_v1.tar
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar
pip install gdown
gdown https://drive.google.com/uc?id=13siMs0HoU3WHkeGvNJxVFOF68BAQedmT -O rugpt3_models.tar.gz
tar -xzvf rugpt3_models.tar.gz && rm rugpt3_models.tar.gz
python -m nltk.downloader 'punkt'