# Multi Paragraph Machine Reading Comprehension with Hybrid Reader over Tables and Text

File Directory

```bash
├── BigBird
│   ├── MRC_bigbird.py
│   ├── MRC_Table_bigbird.py
│   └── MRC_Table_bigbird_adapter.py
├── data
│   ├── get_bigbird_dataset.py
│   └── get_ws_Data.py
├── models
│   ├── BERT_QA.py
│   ├── BERT_T_Adapter.py
│   ├── BERT_T_Adapter3.py
│   ├── BERT_T_Adapter_Stage2.py
│   ├── Dual_Encoder.py
│   └── TAPAS_with_TMN.py
├── pretrain
│   └── BERT_pretrain_adapter.py
└──  utils
    ├── attention_utils.py
    ├── Chuncker.py
    ├── DataHolder_*.py
    ├── Table_Holder.py
    ├── evalutate2.py
    ├── HTML_*.py
    ├── modeling_*.py
    ├── Ranking_ids.py
    ├── tokenization.py
    └── utils.py
``` 
---
## Model Arcitecture

#### Architecture of (a) Separated Reader Model, (b) Our Hybrid Reader Model
![image](https://user-images.githubusercontent.com/64192139/212304681-038ecba6-d8d9-48b2-88fd-95075c5f0a31.png)

#### Arcitecture of Hybrid Reader with projection
![Picture4-1](https://user-images.githubusercontent.com/64192139/212303898-cfa2d7b7-fba4-4300-b549-80f2f3338f40.png)
---
## Data Preparation
- KorQuAD 2.0 (https://korquad.github.io/)
---
## Requirements

Please install the following library requirements specified in the requirements.txt first.
```bash
pip install -r requirements.txt
```
