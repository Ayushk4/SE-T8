import time
from params import params
import torch
from dataloader import MEDataset, METestDataset
from transformers import AutoModel, AutoTokenizer
import torch, torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
import os
torch.manual_seed(params.torch_seed)

#### Data paths ####
basepath = "/home/bt1/17CS10029/measeval/MeasEval/data/"

train_doc_path = basepath + "train/text/"
trial_doc_path = basepath + "trial/txt/"
eval_doc_path = basepath + "eval/text/"

train_label_path = basepath + "train/tsv/"
trial_label_path = basepath + "trial/tsv/"

#### Train one epoch ####
def train(model, dataset, criterion):
    model.train() # Put the model to train

    # Lets keep track of the losses at each update
    train_losses = []
    num_batch = 0

    for batch in dataset.batched_dataset:
        # Unpack the batch
        (texts, labels, att_masks, doc_ids, offsets, token_offsets) = batch

        # Make predictions on the model
        preds = model(texts, att_masks)

        # Take into account padded while calculating loss
        loss_unreduced = criterion(preds.permute(0,2,1), labels)
        loss = (loss_unreduced * att_masks).sum() / (att_masks).sum()

        # Update model weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if num_batch % 10 == 0:
            print("Train loss at {}:".format(num_batch), loss.item())

        num_batch += 1
        # Append losses
        train_losses.append(loss.item())

    return np.average(train_losses)

def evaluate(model, dataset, criterion):
    target_names=["NOT_" + params.task, params.task]

    # Put the model to eval mode
    model.eval()

    # Keep track on predictions
    valid_losses = []
    predicts = []
    gnd_truths = []

    with torch.no_grad():
        for batch in dataset.batched_dataset:
            # Unpack the batch
            (texts, labels, att_masks, doc_ids, offsets, token_offsets) = batch
            # Make predictions on the model
            preds = model(texts, att_masks)

            # Take into account padded while calculating loss
            loss_unreduced = criterion(preds.permute(0,2,1), labels)
            loss = (loss_unreduced * att_masks).sum() / (att_masks).sum()

            # Get argmax of non-padded tokens
            for sent_preds, sent_labels, sent_att_masks in zip(preds, labels, att_masks):
                for token_preds, token_labels, token_masks in zip(sent_preds, sent_labels, sent_att_masks):
                    if token_masks.item() != 0:
                        predicts.append(token_preds.argmax().item())
                        gnd_truths.append((token_labels.item()))
            valid_losses.append(loss.item())

            assert len(predicts) == len(gnd_truths)

    # Create confusion matrix and evaluate on the predictions
    confuse_mat = confusion_matrix(gnd_truths, predicts)
    if params.dummy_run:
        classify_report = None
    else:
        classify_report = classification_report(gnd_truths, predicts,
                                        target_names=target_names,
                                        output_dict=True)

    mean_valid_loss = np.average(valid_losses)
    print("Valid_loss", mean_valid_loss)
    print(confuse_mat)

    if not params.dummy_run:
        for labl in target_names:
            print(labl,"F1-score:", classify_report[labl]["f1-score"])
        print("Accu:", classify_report["accuracy"])
        print("F1-Weighted", classify_report["weighted avg"]["f1-score"])
        print("F1-Avg", classify_report["macro avg"]["f1-score"])

    return mean_valid_loss, confuse_mat ,classify_report


############# Load dataset #############
train_dataset = MEDataset(train_doc_path, train_label_path)
valid_dataset = MEDataset(trial_doc_path, trial_label_path)

if params.dummy_run:
    train_dataset.batched_dataset = train_dataset.batched_dataset[:1]
    valid_dataset = train_dataset

print("Dataset created")
os.system("nvidia-smi")

############# Create model #############

class OurBERTModel(nn.Module):
    def __init__(self):
        super(OurBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(params.bert_type, from_tf=params.from_tf)
        self.drop = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, text, att_mask):
        b, num_tokens = text.shape
        token_type = torch.zeros((b, num_tokens), dtype=torch.long).to(params.device)
        outputs = self.bert(text, attention_mask=att_mask, token_type_ids=token_type)
        return self.classifier(self.drop(outputs['last_hidden_state']))

model = OurBERTModel()
print("Model created")
os.system("nvidia-smi")

print(sum(p.numel() for p in model.parameters()))
model = model.to(params.device)
print("Detected", torch.cuda.device_count(), "GPUs!")
# model = torch.nn.DataParallel(model)

########## Optimizer & Loss ###########

criterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

########## Training loop ###########

for epoch in range(params.n_epochs):
    print("\n\n========= Beginning", epoch+1, "epoch ==========")

    train_loss = train(model, train_dataset, criterion)
    print("\n====EVALUATING On Training set :====\n")
    _, _, _ = evaluate(model, train_dataset, criterion)

    print("\n====EVALUATING On Validation set :====\n")
    valid_loss, confuse_mat, classify_report = evaluate(model, valid_dataset, criterion)

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
    print(print_msg)


## Check Mispredicts 

if not params.dummy_run:
    # Check false positives/negatives
    # Put the model to eval mode and track the predictions
    model.eval()
    valid_losses = []
    correct_sentences = 0
    total_sentences = 0 

    with torch.no_grad():
        for batch in valid_dataset.batched_dataset:
            # Unpack the batch and feed into the model
            (texts, labels, att_masks, doc_ids, offsets, token_offsets) = batch
            preds = model(texts, att_masks)

            # Check for mispredicts
            for sent_preds, sent_labels, sent_att_masks, sent_doc_id, sent_offset in zip(preds, labels, att_masks, doc_ids, offsets):
                this_correct = (sent_preds.argmax(1) == sent_labels).sum()
                this_total = len(sent_labels)

                if this_correct != this_total:
                    print("Mispredict:", sent_doc_id, "for sentence at offset", sent_offset)


# Predict spans on Test, Train and Val set
def predict_spans(dataset, save_path):
    # Put the model to eval mode and track the predictions
    model.eval()

    with torch.no_grad():
        span_dict = {}
        for batch in dataset.batched_dataset:
            (texts, raw_texts, att_masks, doc_ids, offsets, token_offsets) = batch
            preds = model(texts, att_masks)

            for sent_preds, sent_raw_text, sent_att_masks, sent_doc_id, sent_offset, sent_token_offsets in zip(preds, raw_texts, att_masks, doc_ids, offsets, token_offsets):

                this_sentence_positives = []
                curr_positive_idx = -1
                for i, (token_preds, token_labels, token_masks) in enumerate(zip(sent_preds, sent_labels, sent_att_masks)):
                    if token_masks.item() != 0:
                        if token_preds.argmax().item() == 1:
                            if curr_positive_idx == -1:
                                curr_positive_idx = i
                        else:
                            if curr_positive_idx != -1:
                                this_sentence_positives.append([curr_positive_idx, i-1])
                                curr_positive_idx = -1
                    else:
                        if curr_positive_idx != -1:
                            this_sentence_positives.append([curr_positive_idx, i-1])
                            curr_positive_idx = -1
                            break

                # Here convert indices to offsets
                if sent_doc_id not in span_dict.keys():
                    span_dict[sent_doc_id] = {}

                this_sent_spans = []
                for span_offsets in this_sentence_positives:
                    this_sent_spans.append([sent_token_offsets[span_offsets[0]][0],
                                            sent_token_offsets[span_offsets[1]][1]
                                        ])
                
                assert sent_offset[0] not in span_dict[sent_doc_id].keys()
                span_dict[sent_doc_id][sent_offset[0]] = this_sent_spans
    
    json.dump(span_dict, open(save_path, 'w+'))
    
if not params.dummy_run:
    # Save model and predicition
    from datetime import datetime
    folder_name = "/home/bt1/17CS10029/measeval/task1/output/" + params.bert_type.replace('/', '_') #datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    if os.path.isdir(folder_name):
        os.system("rm -rf " + folder_name)
    os.mkdir(folder_name)

    test_dataset = METestDataset(eval_doc_path)

    predict_spans(train_dataset, folder_name+"/train_spans.json")
    predict_spans(valid_dataset, folder_name+"/trial_spans.json")
    predict_spans(test_dataset, folder_name+"/test_spans.json")

    torch.save(model.state_dict(), folder_name+"/model.pt")
    json.dump(vars(params), open(folder_name+"/params.json", 'w+'))

