import sys
import os
import inspect

import torch

this_script_location = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(this_script_location, '..'))
sys.path.append(os.path.join(this_script_location, '..', 'GRACE'))

from GRACE.ate_asc_run import init_model
from GRACE.ate_asc_modeling import BertForSequenceLabeling as GraceASC
from GRACE.ate_modeling import BertForSequenceLabeling as GraceATE
from GRACE.tokenization import BertTokenizer


# Backported from
# https://github.com/ronaldseoh/GRACE/blob/ab32a79677ed6dd5dfcfb14aaa4d6422ff48675c/ate_asc_run.py#L138
def load_grace_model(model_path, ate_or_asc='asc', bert_arch_name='bert-base-uncased',
                     num_labels=3, num_tp_labels=(3,5),
                     at_labels=['O', 'B-AP', 'I-AP'],
                     num_decoder_layer=2, decoder_shared_layer=3,
                     use_ghl=True, use_vat=True,
                     cache_dir=None):

    model_state_dict = torch.load(model_path, map_location='cpu')
    
    if ate_or_asc == 'asc'
        task_config = {
            "use_ghl": use_ghl,
            "use_vat": use_vat,
            "num_decoder_layer": num_decoder_layer,
            "decoder_shared_layer": decoder_shared_layer,
            "at_labels": at_labels
        }

        if num_decoder_layer != 0:
            layer_num_list = [layer_num for layer_num in range(num_decoder_layer)]

            new_model_state_dict = {}

            model_state_dict_exsit_keys = model_state_dict.keys()

            max_bert_layer = max([int(k_str.split(".")[3]) for k_str in model_state_dict_exsit_keys if "bert.encoder.layer" in k_str])

            for k_str in model_state_dict_exsit_keys:
                new_model_state_dict[k_str] = model_state_dict[k_str]

                for layer_num in layer_num_list:
                    bert_key_name = "bert.encoder.layer.{}".format(max_bert_layer - num_decoder_layer + 1 + layer_num)

                    mirror_key_name = "bert.encoder.layer.{}".format(layer_num)

                    if k_str.find(bert_key_name) == 0:
                        new_key_name = k_str.replace(bert_key_name, mirror_key_name).replace("bert.encoder", "decoder.decoder")

                        if "attention.self" in new_key_name:
                            new_key_name_sufx = new_key_name.replace("attention.self", "slf_attn.att")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                            new_key_name_sufx = new_key_name.replace("attention.self", "enc_attn.att")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                        elif "attention.output" in new_key_name:
                            new_key_name_sufx = new_key_name.replace("attention.output", "slf_attn.output")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                            new_key_name_sufx = new_key_name.replace("attention.output", "enc_attn.output")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                        else:
                            new_model_state_dict[new_key_name] = model_state_dict[k_str].clone()

                if k_str.find("bert.embeddings") == 0:
                    new_key_name = k_str.replace("bert.embeddings", "decoder.embeddings")
                    new_model_state_dict[new_key_name] = model_state_dict[k_str].clone()

            model_state_dict = new_model_state_dict

        # Prepare model
        model = GraceASC.from_pretrained(
            bert_arch_name,
            state_dict=model_state_dict, cache_dir=cache_dir,
            num_tp_labels=num_tp_labels, task_config=task_config)
            
    elif ate_or_asc == 'ate':

        task_config = {
            "use_ghl": use_ghl,
            "use_vat": use_vat,
            "num_decoder_layer": num_decoder_layer,
            "decoder_shared_layer": decoder_shared_layer,
            "at_labels": at_labels
        }
        
        if "model_state_dict" in model_state_dict:
            model_state_dict = model_state_dict['model_state_dict']
            
        model = GraceATE.from_pretrained(
            bert_arch_name,
            state_dict=model_state_dict, cache_dir=cache_dir,
            num_labels=num_labels, task_config=task_config)

    return model

def load_grace_tokenizer(bert_arch_name='bert-base-uncased',
                         do_lower_case=True, cache_dir=None):

    tokenizer = BertTokenizer.from_pretrained(
        bert_arch_name, cache_dir=cache_dir,
        do_lower_case=do_lower_case)
    
    return tokenizer
