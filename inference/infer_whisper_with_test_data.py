import argparse
import csv

from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset, Audio
import evaluate
from unicode_tr import unicode_tr

wer_metric = evaluate.load("wer")


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return unicode_tr(sample["text"]).lower()
    elif "sentence" in sample:
        return unicode_tr(sample["sentence"]).lower()
    elif "normalized_text" in sample:
        return unicode_tr(sample["normalized_text"]).lower()
    elif "transcript" in sample:
        return unicode_tr(sample["transcript"]).lower()
    elif "transcription" in sample:
        return unicode_tr(sample["transcription"]).lower()
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


whisper_norm = BasicTextNormalizer()


def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}


def main(args):
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=args.model_id, device=args.device,
        use_auth_token="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"
    )
    dataset = load_dataset("json", data_files=args.data_json)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    predictions = []
    references = []

    # run streamed inference
    for out in whisper_asr(data(dataset), batch_size=args.batch_size):
        predictions.append(whisper_norm(out["text"]))
        references.append(out["reference"][0])
        print(whisper_norm(out["text"]), out["reference"][0])
    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    print("WER:", wer)

    if args.result is not None:
        f_o = open(args.result, 'w', encoding='utf-8', newline='')
        writer = csv.writer(f_o)
        header = ['ref', 'hyp']
        writer.writerow(header)
        for ref, hyp in zip(references, predictions):
            writer.writerow([ref, hyp])
        writer.writerow(f"wer: {wer}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_json",
        type=str,
        required=True,
        help="Data json file for test. Format: audio: /path, text: transcription.",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )

    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Two letter language code for the transcription language, e.g. use 'en' for English.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--result",
        type=str,
        help="Result csv file that contains ref and hypothesis.",
    )

    args = parser.parse_args()

    main(args)
