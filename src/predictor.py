import torch
from model import LOTClassModel
import os
from transformers import BertTokenizer

class LOTClassPredictor(object):

    def __init__(self, dataset_dir, model_file='final_model.pt', label_names_file='label_names.txt', max_len=512):
        self.pretrained_lm = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        label_name_file = open(os.path.join(dataset_dir, label_names_file))
        self.label_names = [line.strip() for line in label_name_file.readlines()]
        num_class = len(self.label_names)
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   num_labels=num_class)

        loader_file = os.path.join(dataset_dir, model_file)
        assert os.path.exists(loader_file)
        print(f"\nLoading model from {loader_file}")
        self.model.load_state_dict(torch.load(loader_file))
        self.max_len = max_len

    def __encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=self.max_len,
                                                        padding='max_length',
                                                        return_attention_mask=True, truncation=True,
                                                        return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    def predict(self, docs):
        """ predict the labels for a list of plain text documents

        :param docs: python list of str storing the input documents
        :return: a list of predicted class ids
        """
        #tokenize and get the input_id and mask
        input_ids, attention_masks = self.__encode(docs)

        #predict
        with torch.no_grad():
            logits = self.model(input_ids,
                           pred_mode="classification",
                           token_type_ids=None,
                           attention_mask=attention_masks)
            logits = logits[:, 0, :]
            pred_labels = torch.argmax(logits, dim=-1).cpu()
            return pred_labels


if __name__ == "__main__":
    predictor = LOTClassPredictor('datasets/dbpedia', model_file='temp_final_model.pt', label_names_file='label_names.txt', max_len=200)
    docs = ["The best singer in the world has a record performance.", "Olympic medal winner plays tennis like a pro."]
    preds = predictor.predict(docs)
    print(preds)
    print([predictor.label_names[i] for i in preds])
