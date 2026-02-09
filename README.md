# LLAUREC
pytorch implementation for "**CoLDRec: Large Language Model Embeddings with Collaborative Filtering-Guided Diffusion for Recommendation**"
<img width="1651" height="1545" alt="pipeline2" src="https://github.com/user-attachments/assets/6051bfb4-4b8d-48d3-9c33-e4f96d2d6974" />

```
VIRAL/
│
├── data/
│   └── <dataset_name>/
│       ├── train/test/valid_mat.pkl # train/valid/test set (sparse matrix)
│       ├── <dataset_name>_asin.json
│       ├── sample_user_profile.json
│       ├── meta_<dataset_name>.json # contains meta data
│       └── 
├── README.md
├── src
│	├──get_text_feat.py              # Script to generate text embeddings
│	├──README.md
│   └─requirements.txt
```

Convert dataset preproced in RLM [paper](https://arxiv.org/abs/2310.15950) to our standard.

```sh train/valid/test to <dataset_name>.inter
python src/convert.py -d book
```



```sh convert item infor to embeding
python src/preprocess.py -d book
```


```sh tuning and generate profile for user
python src/tuning.py -d book -t False
python src/generate_profile.py -d book -t True
```


```sh convert profile user to embedding
python src/preprocess.py -d book -u True -t True
```

Run
```sh
python src/main.py -d book
```
