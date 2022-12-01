import pathlib
import itertools
import argparse
import logging

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer


logging.basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name, model_path, device="cuda"):
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    return model

def create_erased_inputs(x, s, cls_explainer, tokenizer):
    word_attributions = cls_explainer(s)

    erased = list()
    mod_input_ids = x.input_ids[0].clone()

    word_attributions = [(w, sc) for w, sc in word_attributions if w != ""]

    if len(word_attributions) != mod_input_ids.shape[0]:
        return None

    for i, _ in sorted(enumerate(word_attributions), key=lambda x: x[1][1]):
        if i == 0 or i == x.input_ids[0].shape[0]-1:
            continue

        mod_input_ids[i] = tokenizer.mask_token_id
        erased.append(mod_input_ids.clone())
    
    return torch.stack(erased[:-1])

def enumerate_hypothesis(s, cls_explainer, model, tokenizer):
    x = tokenizer(s, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model(input_ids=x.input_ids)
        origin_logits = output.logits[0].cpu().numpy()

        erased_inputs = create_erased_inputs(x, s, cls_explainer, tokenizer)

        if erased_inputs is None:
            return None

        output = model(input_ids=erased_inputs)

    predictions = list()

    for logit, masked_x, s in zip(output.logits, erased_inputs, tokenizer.batch_decode(erased_inputs)):
        logit = logit.cpu().softmax(dim=0).numpy()

        predictions.append((s,
                            len([t for t in masked_x if t == tokenizer.mask_token_id]),
                            origin_logits.argmax() == logit.argmax(),
                            origin_logits.max() - logit.max(),
                            origin_logits.argmax().item(),
                            logit.argmax(0).item(),
                            ))
        
    predictions.sort(key=lambda x: (-int(x[2]), -x[1], -x[3]))
    return predictions

def main(args):
    model = load_model(args.model_name, args.model_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer)

    s = "I do not hate this movie."

    for p in enumerate_hypothesis(s, cls_explainer, model, tokenizer):
        print(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent
    args.model_path = BASE_DIR / 'output' / 'SA' / 'epoch=3'
    main(args)
