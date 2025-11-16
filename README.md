This is the implementation of MultiRankDTA: A Hybrid Ranking Framework for Robust and Generalizable Drug–Target Affinity Prediction
1. Overview
   Overview of MultiRankDTA. (a) Base Model and Joint Representation Learning. Drug molecules are transformed into molecular graphs using RDKit and encoded via a multi-layer MPNN to capture structural features. Protein sequences are encoded by the pretrained ESM-2 model, yield contextualized residue embeddings that represent global semantic information. The pooled embeddings are fused into a joint representation and passed through a self-attention module to model cross-modal interactions. This representation is then fed into three ranking experts—pointwise, pairwise, and listwise—to learn numerical affinity values, relative preference, and global ranking structure, respectively. The outputs are integrated via a gating-based mixture-of-experts mechanism to produce the final affinity prediction y. (b) Multi-Source Learning. To enhance generalization across distribution shifts, each sample is independently passed through base models trained on Davis, KIBA, and BindingDB, from which we extract their shared representations and prediction outputs. These vectors are concatenated into a unified high-dimensional representation and fed into a lightweight MetaMLP, improving the predictive performance on the target task.
<img width="433" height="178" alt="image" src="https://github.com/user-attachments/assets/5a206855-8743-40a9-ad57-84b729d03e3f" />

2.Project Structure
MultiRankDTA/
├── code/
│   ├── changetransformer.py   # Main model (TransformerModel)
│   ├── create_data.py         # Data preprocessing scripts
│   ├── graph.py               # SMILES to molecular graphs
│   ├── listwise_loss.py       # rank loss functions
│   ├── utils.py               # Metrics and utility functions
│   ├── trynew.py              # multi-Source Learning
│   ├── trytaskduddavis.py     # Single-dataset training & testing
├── data/
│   ├── davis
│   ├── kiba
│   ├── Bindingdb/
│   │   ├── bindingdbIC_2train.csv
│   │   └── bindingdbIC_2test.csv
│   ├── davis_2train.csv
│   ├── davis_2test.csv
│   ├── kiba_2train.csv
│   ├── kiba_2test.csv

└── README.md

