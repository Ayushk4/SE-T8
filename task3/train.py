import time
from params import params
import torch
from dataloader import MEDataset#, METestDataset
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
def train(model, dataset, criterion, epoch_num):
    # Put the model to train
    model.train()

    # Lets keep track of the losses at each update
    train_losses = []
    num_batch = 0

    loss_scaling = {'me': 1.0, 'mp': 1.0, 'qual': 1.0}
    print("Loss Scale:", loss_scaling)
    
    for batch in dataset.batched_dataset:
        # Unpack the batch
        (texts, labels, att_masks, _, _, _, _) = batch

        # Make predictions on the model
        if not params.token_type_ids_not:
            preds = model(texts, att_masks, labels['QUANT'])
        else:
            preds = model(texts, att_masks)

        # Take into account padded while calculating loss
        loss = 0
        for single_task, single_task_preds in preds.items():
            loss_unreduced = criterion(single_task_preds.permute(0,2,1), labels[single_task.upper()])
            loss += ((loss_unreduced * att_masks).sum() / (att_masks).sum()) * loss_scaling[single_task]
        loss.backward()

        # Update model weights
        optimizer.step()
        optimizer.zero_grad()

        if num_batch % 10 == 0:
            print("Train loss at {}:".format(num_batch), loss.item())

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

def evaluate(model, dataset, criterion, epoch_num):
    # Put the model to eval mode
    model.eval()

    # Keep track of predictions
    valid_losses = []
    predicts, gnd_truths = {}, {}
    if params.me:
        predicts['me'] = []
        gnd_truths['me'] = []
    if params.mp:
        predicts['mp'] = []
        gnd_truths['mp'] = []
    if params.qual:
        predicts['qual'] = []
        gnd_truths['qual'] = []

    loss_scaling = {'me': 1.0, 'mp': 1.0, 'qual': 1.0}
    print("Loss Scale:", loss_scaling)

    with torch.no_grad():
        for batch in dataset.batched_dataset:
            # Unpack the batch
            (texts, labels, att_masks, _, _, _, _) = batch
            # Make predictions on the model
            if not params.token_type_ids_not:
                preds = model(texts, att_masks, labels['QUANT'])
            else:
                preds = model(texts, att_masks)

            # Take into account padded while calculating loss
            loss = 0
            for single_task, single_task_preds in preds.items():
                loss_unreduced = criterion(single_task_preds.permute(0,2,1), labels[single_task.upper()])
                loss += ((loss_unreduced * att_masks).sum() / (att_masks).sum()) * loss_scaling[single_task]

            assert preds.keys() == predicts.keys()
            assert preds.keys() == gnd_truths.keys()
            
            # Get argmax of non-padded tokens
            for task in preds.keys():
                for sent_preds, sent_labels, sent_att_masks in zip(preds[task], labels[task.upper()], att_masks):
                    for token_preds, token_labels, token_masks in zip(sent_preds, sent_labels, sent_att_masks):
                        if token_masks.item() != 0:
                            predicts[task].append(token_preds.argmax().item())
                            gnd_truths[task].append((token_labels.item()))
            valid_losses.append(loss.item())

            assert len(predicts) == len(gnd_truths)

    # Create confusion matrix and evaluate on the predictions
    for task in preds.keys():
        print("\nTask:", task)
        target_names = ["NOT_" + task.upper(), task.upper()]
        
        confuse_mat = confusion_matrix(gnd_truths[task], predicts[task])
        if params.dummy_run:
            classify_report = None
        else:
            classify_report = classification_report(gnd_truths[task], predicts[task],
                                            target_names=target_names,
                                            output_dict=True)

        print(confuse_mat)

        if not params.dummy_run:
            for labl in target_names:
                print(labl, "F1-score:", classify_report[labl]["f1-score"])
            print("Accu:", classify_report["accuracy"])
            print("F1-Weighted", classify_report["weighted avg"]["f1-score"])
            print("F1-Avg", classify_report["macro avg"]["f1-score"])

    mean_valid_loss = np.average(valid_losses)
    print("\nValidation loss", mean_valid_loss)

    return mean_valid_loss


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
        self.bert = AutoModel.from_pretrained(params.bert_type)
        self.drop = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.me_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        if params.me:
            self.non_lin = nn.Softmax(2)
            self.mp_classifier = nn.Linear(self.bert.config.hidden_size+3, 2)
            self.qual_classifier = nn.Linear(self.bert.config.hidden_size+3, 2)
        else:
            self.mp_classifier = nn.Linear(self.bert.config.hidden_size, 2)
            self.qual_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        

    def forward(self, text, att_mask, token_type=None):
        b, num_tokens = text.shape
        if token_type == None:
            assert params.token_type_ids_not
            token_type = torch.zeros((b, num_tokens), dtype=torch.long).to(params.device)
        
        if "roberta" in params.bert_type:
            outputs = self.bert(text, attention_mask=att_mask)
        else:
            outputs = self.bert(text, attention_mask=att_mask, token_type_ids=token_type)
        classifier_in = self.drop(outputs['last_hidden_state'])

        tags = {}
        if params.me:
            tags['me'] = self.me_classifier(classifier_in)
            softmaxed = self.non_lin(tags['me'])
            # Extract features from logits of me and concatenate
            mean_feats = (att_mask.unsqueeze(-1) * softmaxed).sum(1)/att_mask.sum(1).unsqueeze(-1)
            max_feats = (softmaxed[:, :, 1].max(1)[0]).unsqueeze(-1)
            extra_feats = torch.cat([mean_feats, max_feats], -1).unsqueeze(1).expand(-1, num_tokens, -1)
            
            classifier_in = torch.cat([classifier_in, extra_feats], -1)

        if params.mp:
            tags['mp'] = self.mp_classifier(classifier_in)
        if params.qual:
            tags['qual'] = self.qual_classifier(classifier_in)

        return tags

model = OurBERTModel()
print("Model created")
os.system("nvidia-smi")

model.bert.resize_token_embeddings(len(train_dataset.bert_tok))
# print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
# embedding_size = model.bert.embeddings.word_embeddings.weight.size(1)
# new_embeddings = torch.FloatTensor(2, embedding_size).uniform_(-0.1, 0.1)
# print("new_embeddings shape:", new_embeddings.size())
# new_embedding_weight = torch.cat((model.bert.embeddings.word_embeddings.weight.data,new_embeddings), 0)
# model.bert.embeddings.word_embeddings.weight.data = new_embedding_weight
# print("Updated Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
# Update model config vocab size
model.bert.config.vocab_size = model.bert.config.vocab_size + 2

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

    train_loss = train(model, train_dataset, criterion, epoch)
    # print("\n====EVALUATING On Training set :====\n")
    # _ = evaluate(model, train_dataset, criterion, epoch)

    print("\n====EVALUATING On Validation set :====\n")
    valid_loss = evaluate(model, valid_dataset, criterion, epoch)

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
    print("\n", print_msg, "\n")


## Predict on Test set

def predict_spans(dataset, save_path):
    model.eval()
    predicts = {}

    with torch.no_grad():
        all_tasks = []
        if params.me: 
            all_tasks.append('me')
        if params.mp: 
            all_tasks.append('mp')
        if params.qual: 
            all_tasks.append('qual')
        
        for task_type in all_tasks:
            this_labels_added = []
            for batch in dataset.batched_dataset:
                (texts, token_type_ids, att_masks, token_offsets, quant_data) = batch
                if not params.token_type_ids_not:
                    preds = model(texts, att_masks, token_type_ids['QUANT'])
                else:
                    preds = model(texts, att_masks)

                
                preds = preds[task_type]
                for sent_preds, sent_att_masks, sent_token_offsets, sent_quant_data in zip(preds, att_masks, token_offsets, quant_data):

                    this_sentence_positives = []
                    curr_positive_idx = -1
                    for i, (token_preds, token_masks) in enumerate(zip(sent_preds, sent_att_masks)):
                        if token_masks.item() != 0:
                            if token_preds.argmax().item() == 1:
                                if curr_positive_idx == -1:
                                    curr_positive_idx = i
                            else:
                                if curr_positive_idx != -1:
                                    this_sentence_positives.append([curr_positive_idx, i-1])
                                    curr_positive_idx = -1
                                    break
                        else:
                            if curr_positive_idx != -1:
                                this_sentence_positives.append([curr_positive_idx, i-1])
                                curr_positive_idx = -1
                                break

                    # Here convert indices to offsets
                    # assert len(this_sentence_positives) == 1
                    if len(this_sentence_positives) != 0:
                        span_offsets = this_sentence_positives[0]
                        this_label_offs = (sent_token_offsets[span_offsets[0]][0], sent_token_offsets[span_offsets[1]][1])
                        this_labels_added.append([sent_quant_data, (task_type, this_label_offs )])

            json.dump(this_labels_added, open(save_path + '_' + task_type + ".json", 'w+'))

if not params.dummy_run:
    from testloader import METestDataset
    from datetime import datetime
    folder_name = "/home/bt1/17CS10029/measeval/task3/output/" + params.bert_type.replace('/', '_')  # "_" + datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    if os.path.isdir(folder_name):
        os.system("rm -rf " + folder_name)
    os.mkdir(folder_name)

    predict_train_dataset = METestDataset(train_doc_path, params.test_label_folder + "/train_labels.json")
    predict_spans(predict_train_dataset, folder_name + "/train_spans")

    predict_trial_dataset = METestDataset(trial_doc_path, params.test_label_folder + "/trial_labels.json")
    predict_spans(predict_trial_dataset, folder_name + "/trial_spans")
    
    predict_test_dataset = METestDataset(eval_doc_path, params.test_label_folder + "/test_labels.json")
    predict_spans(predict_test_dataset, folder_name + "/test_spans")

    torch.save(model.state_dict(), folder_name+"/model.pt")
    json.dump(vars(params), open(folder_name+"/params.json", 'w+'))


