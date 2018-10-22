# BERT-pytorch
PyTorch implementation of BERT in "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (https://arxiv.org/abs/1810.04805)

## Requirements
- Python 3.6+
- [PyTorch 4.1+](http://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)

All dependencies can be installed via:

```
pip install -r requirements.txt
```

## Quickstart

### Prepare data
First things first, you need to prepare your data in an appropriate format. 
Your corpus is assumed to follow the below constraints.

- Each line is a *document*.
- A *document* consists of *sentences*, seperated by vertical bar (|).
- A *sentence* is assumed to be already tokenized. Tokens are seperated by space.
- A *sentence* has no more than 256 tokens.
- A *document* has at least 2 sentences. 
- You have two distinct data files, one for train data and the other for val data.

This repo comes with example data for pretraining in data/example directory.
Here is the content of data/example/train.txt file.

```
One, two, three, four, five,|Once I caught a fish alive,|Six, seven, eight, nine, ten,|Then I let go again.
I’m a little teapot|Short and stout|Here is my handle|Here is my spout.
Jack and Jill went up the hill|To fetch a pail of water.|Jack fell down and broke his crown,|And Jill came tumbling after.  
```

Also, this repo includes SST-2 data in data/SST-2 directory for sentiment classification.

### Build dictionary
```
python bert.py preprocess-index data/example/train.txt --dictionary=dictionary.txt
```
Running the above command produces dictionary.txt file in your current directory.

### Pre-train the model
```
python bert.py pretrain --train_data data/example/train.txt --val_data data/example/val.txt --checkpoint_output model.pth
```
This step trains BERT model with unsupervised objective. Also this step does:
- logs the training procedure for every epoch
- outputs model checkpoint periodically
- reports the best checkpoint based on validation metric

### Fine-tune the model
You can fine-tune pretrained BERT model with downstream task.
For example, you can fine-tune your model with SST-2 sentiment classification task. 
```
python bert.py finetune --pretrained_checkpoint model.pth --train_data data/SST-2/train.tsv --val_data data/SST-2/dev.tsv
```
This command also logs the procedure, outputs checkpoint, and reports the best checkpoint.

## See also
- [Transformer-pytorch](https://github.com/dreamgonfly/Transformer-pytorch) : My own implementation of Transformer. This BERT implementation is based on this repo.

## Author
[@dreamgonfly](https://github.com/dreamgonfly)