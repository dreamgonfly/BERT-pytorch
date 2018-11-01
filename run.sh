wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 --directory-prefix data/wiki/
python main.py preprocess-all --data_dir data/wiki