# CoLDRec
pytorch implementation for "**CoLDRec: Large Language Model Embeddings with Collaborative Filtering-Guided Diffusion for Recommendation**"
<img width="3040" height="1527" alt="Picture4" src="https://github.com/user-attachments/assets/211fbb68-bee4-4eb1-9d45-8b6e414f2a9d" />

```
/
│
├── data/
│   └── <dataset_name>/
│       ├── <dataset_name>.inter # interaction file
│       ├── item_feat.npy # llm generated profile for items (based on reviews)
│       ├── user_feat.npy # llm generated profiles for each user
│       └── 
├── README.md
├── src
│	├──main.py              
│   └─requirements.txt
```

Run
```sh
python src/main.py -d book
```
