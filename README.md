
This is the official repo for **Multi Paragraph Machine Reading Comprehension with Hybrid Reader over Tables and Text**

## Abstract
In machine reading comprehension, the answer to a question could be in a table or text. Previous studies proposed combining specialized models for each table and text. Instead, we propose a Hybrid Reader Model for both the table and text with a modified K-Adapter. In the K-Adapter, the adapters are trained in a distributed manner with their weights fixed in each pre-training stage. This training process infuses table knowledge into the pre-trained model while retaining their original weights from pre-training. With a modified K-Adapter and BERT as the backbone model, our Hybrid Reader Model performs better than the specialized model on the Korean MRC dataset KorQuad 2.0.

## Model Arcitecture

#### Architecture of (a) Separated Reader Model, (b) Our Hybrid Reader Model
![image](https://github.com/delosyCho/Multi-Paragraph-Machine-Reading-Comprehension-with-Hybrid-Reader/assets/64192139/7efff684-9ade-4b00-ac73-c52bad7d4c16)


#### Arcitecture of Hybrid Reader with projection
<center><img src="https://user-images.githubusercontent.com/64192139/212303898-cfa2d7b7-fba4-4300-b549-80f2f3338f40.png" width="70%" height="70%"></center>


## File Directory

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
├── pretrain
│   ├── modeling_adapter.py
│   ├── attention_utils.py
│   ├── modelings.py
│   └── trainer.py
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


## Data Preparation
KorQuAD 2.0 (https://korquad.github.io/)

## Requirements

Please install the following library requirements specified in the requirements.txt first.
If you want to download all library at once, use this code.
```bash
pip install -r requirements.txt
```

## License
