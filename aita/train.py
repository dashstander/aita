from aita.aita_dataset import AitaDataset
import datetime
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import torch
from torch.utils.data import DataLoader, random_split, SequentialSampler, RandomSampler





def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def pad_data(tokenizer, input_ids):
    # We'll borrow the `pad_sequences` utility function to do this.

    MAX_LEN = 512

    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(
        input_ids,
        maxlen=MAX_LEN,
        dtype="long", 
        value=0,
        truncating="post",
        padding="post"
    )
    print('\nDone')
    return input_ids


def set_up_datasets():
     # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    aita_dataset = AitaDataset(tokenizer, max_length=512)

    train_size = int(np.floor(len(aita_dataset) * 0.9))
    test_size = len(aita_dataset) - train_size
    batch_size = 32


    # Create the DataLoader for our training set.
    train_data, validation_data = random_split(aita_dataset, [train_size, test_size])
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=4)

    # Create the DataLoader for our validation set.
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


def train(train_dataloader, epochs=4):
    device = torch.device('cpu')

    print("Loading model")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 1, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        max_position_embeddings=512
    )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(
        model.parameters(),
        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
    )

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, # Default value in run_glue.py
        num_training_steps=total_steps
    )

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


    # Set the seed value all over the place to make this reproducible.
    # seed_val = 42

    # random.seed(seed_val)
    # np.random.seed(seed_val)
    # torch.manual_seed(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):
    
        # ========================================
        #               Training
        # ========================================
    
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):



            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            if step > 1:
                break

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(
                b_input_ids, 
                # token_type_ids=None, 
                attention_mask=b_input_mask, 
                labels=b_labels
            )
        
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)            
    
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

        print("")
        print("\tAverage training loss: {0:.2f}".format(avg_train_loss))
        print("\tTraining epoch took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Training complete!")

    return model


def validate(validation_dataloader, model):
    # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for i, batch in enumerate(validation_dataloader):
            
            if i > 2:
                break
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
        
            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask
                )
        
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.numpy()
            label_ids = b_labels.numpy()
        
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("\tAccuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("\tValidation took: {:}".format(format_time(time.time() - t0)))


def save_model(model, path):
    model.save_pretrained(path)


def main():
    train_loader, valid_loader = set_up_datasets()

    model = train(train_loader, epochs=1)

    validate(valid_loader, model)

    save_model(model, '/Users/dashiell/workspace/aita/models')

if __name__ == '__main__':
    main()
