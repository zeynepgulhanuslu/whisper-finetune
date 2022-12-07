import argparse
import os
import re

import evaluate
from datasets import Audio, load_from_disk
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer

from dataloader.convert_kaldi_data import get_dataset

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\'\‘\”\�\…\{\}\【\】\・\。\『\』\、\ー\〜]'  # remove special character tokens

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch


def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[â]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[î]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ô]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[û]', 'u', batch["sentence"])
    batch["sentence"] = re.sub('[é]', 'e', batch["sentence"])
    batch["sentence"] = re.sub('[é]', 'e', batch["sentence"])
    return batch


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"],
                                                sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, required=True, help='train csv data file')
    parser.add_argument('--test', type=str, required=True, help='test csv data file')
    parser.add_argument('--num_proc', type=int, required=True, help='num process counts')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--out_dir', type=str, required=True, help='output directory')
    parser.add_argument('--save_feats', type=bool, default=False, help='save feature')

    args = parser.parse_args()
    train_file = args.train
    test_file = args.test

    num_process = args.num_proc
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('creating test dataset')
    test_dataset = get_dataset(test_file)
    test_dataset = test_dataset.map(replace_hatted_characters)
    test_dataset = test_dataset.map(remove_special_characters)
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    print('creating train dataset')
    train_dataset = get_dataset(train_file)
    train_dataset = train_dataset.map(replace_hatted_characters)
    train_dataset = train_dataset.map(remove_special_characters)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Turkish", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Turkish", task="transcribe")

    if args.save_feats:

        test_data_dir = os.path.join(out_dir, 'test-data')
        train_data_dir = os.path.join(out_dir, 'train-data')

        if not os.path.exists(test_data_dir):
            print('preparing test dataset as batches')
            test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names,
                                            num_proc=num_process, keep_in_memory=True)

            test_dataset.save_to_disk(test_data_dir)
        else:
            print('loading test dataset as batches')
            test_dataset = load_from_disk(test_data_dir)
        if not os.path.exists(train_data_dir):
            print('preparing train dataset as batches')
            train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names,
                                              num_proc=num_process, keep_in_memory=True)

            train_dataset.save_to_disk(train_data_dir)
        else:
            print('loading train dataset as batches')
            train_dataset = load_from_disk(train_data_dir)
    else:

        test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names,
                                        num_proc=num_process)
        print('creating training dataset')
        train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names,
                                          num_proc=num_process)
        print('done creating train dataset')

    print('batch dataset completed.')

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    print('whisper small model loaded.')
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,  # change to a repo name of your choice
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        optim="adamw_bnb_8bit",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    print('saving pretrained model')
    processor.save_pretrained(out_dir)
    print('training started')
    trainer.train()
    print('training finished')
