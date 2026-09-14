import argparse
import cv2
import numpy as np
import onnxruntime

"""Tokenization classes."""
import collections
import re
import unicodedata
import six
from functools import lru_cache
import os
import numpy as np


@lru_cache()
def default_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.txt")


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file=default_vocab(), do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    @staticmethod
    def convert_tokens_to_string(tokens, clean_up_tokenization_spaces=True):
        """ Converts a sequence of tokens (string) in a single string. """

        def clean_up_tokenization(out_string):
            """ Clean up a list of simple English tokenization artifacts
            like spaces before punctuations and abreviated forms.
            """
            out_string = (
                out_string.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
            )
            return out_string

        text = ' '.join(tokens).replace(' ##', '').strip()
        if clean_up_tokenization_spaces:
            clean_text = clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def vocab_size(self):
        return len(self.vocab)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


###使用 tokenizer 标记器, 将文本数据转换成 input_ids, token_type_ids, attention_mask 数据. token_type_ids 类型标记的作用是告诉模型输入的是第几句, 单句则全是 0, 在批量转换时, 不同句值不同.
###https://juejin.cn/post/7266344040760590376
###https://zhuanlan.zhihu.com/p/341994096
###padding用于填充。它的参数可以是布尔值或字符串：(1):True或”longest“：填充到最长序列（如果你仅提供单个序列，则不会填充）.“max_length”：用于指定你想要填充的最大长度，如果max_length=Flase，那么填充到模型能接受的最大长度（这样即使你只输入单个序列，那么也会被填充到指定长度）. (2):False或“do_not_pad”：不填充序列。如前所述，这是默认行为。
def tokenize(_tokenizer, texts, specical_texts, context_length: int = 52):
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    input_ids = np.array(all_tokens, dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)  ###单句则全是 0
    attention_mask = (input_ids > 0).astype(np.bool_)

    specical_tokens = []
    for text in specical_texts:
        specical_tokens.append(_tokenizer.vocab[text])

    return input_ids, token_type_ids, attention_mask, specical_tokens


def generate_masks_with_special_tokens_and_transfer_map(input_ids, special_tokens_list):
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token)).astype(np.bool_)  ###高版本的numpy需要np.bool_
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.array(np.nonzero(special_tokens_mask)).T

    # generate attention mask and positional ids
    text_self_attention_masks = np.tile(np.expand_dims(np.eye(num_token, dtype=np.bool_), axis=0), (bs, 1, 1))
    position_ids = np.zeros((bs, num_token))
    # cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            text_self_attention_masks[row, col, col] = True
            position_ids[row, col] = 0
        else:
            text_self_attention_masks[row, previous_col + 1: col + 1, previous_col + 1: col + 1] = True
            position_ids[row, previous_col + 1: col + 1] = np.arange(0, col - previous_col)
            # c2t_maski = np.zeros((num_token)).astype(np.bool_)
            # c2t_maski[previous_col + 1 : col] = True
            # cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col
    return text_self_attention_masks, position_ids.astype(np.int64)

def resize_image(srcimg, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):  ###返回(高度, 宽度)
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
    size = get_size((img.shape[1], img.shape[0]), size, max_size)
    rescaled_image = cv2.resize(img, (size[1], size[0]))
    return rescaled_image


def get_phrases_from_posmap(posmap, input_ids, tokenizer, left_idx=0, right_idx=255):
    if posmap.ndim == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = np.nonzero(posmap)[0].tolist()
        if len(non_zero_idx) > 0:
            token_ids = [input_ids[i] for i in non_zero_idx]
            return tokenizer.convert_ids_to_tokens(token_ids)[0]
        else:
            return None
    else:
        raise NotImplementedError("posmap must be 1-dim")


class GroundingDINO():
    def __init__(self, modelpath, box_threshold, vocab_path,device, text_threshold=None, with_logits=True):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 4
        if device == "cpu":
            self.net = onnxruntime.InferenceSession(modelpath, so, providers=['CPUExecutionProvider'])
        elif device == "cuda":
            self.net = onnxruntime.InferenceSession(modelpath, so, providers=['CUDAExecutionProvider'])
        else:
            self.errorMessage(
                self.tr("Invalid providers"),
                self.tr("Invalid infer providers '{}'").format(
                    self._config["device"]
                ),
            )
            return False
        self.input_names = ["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"]
        self.output_names = ["logits", "boxes"]
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.with_logits = with_logits

        self.size = [1200, 800]  ###(宽度, 高度)
        self.max_size = None
        # self.size = 800
        # self.max_size = 1333
        self.mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        self.std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
        self.max_text_len = 256
        self.specical_texts = ["[CLS]", "[SEP]", ".", "?"]
        self.tokenizer = FullTokenizer(vocab_file=vocab_path)

    def detect(self, srcimg, text_prompt, size=None, box_threshold=None, nms_threshold=0.7):
        infer_size = size if size is not None else self.size
        infer_box_threshold = self.box_threshold if box_threshold is None else box_threshold
        rescaled_image = resize_image(srcimg, infer_size, max_size=self.max_size)
        img = (rescaled_image.astype(np.float32) / 255.0 - self.mean_) / self.std_
        inputs = {"img": np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)}
        import time

        start_time = time.perf_counter()
        caption = text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + " ."

        input_ids, token_type_ids, attention_mask, specical_tokens = tokenize(self.tokenizer, caption,
                                                                              self.specical_texts,
                                                                              context_length=self.max_text_len)
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids,
                                                                                                      specical_tokens)
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]

            position_ids = position_ids[:, : self.max_text_len]
            input_ids = input_ids[:, : self.max_text_len]
            attention_mask = attention_mask[:, : self.max_text_len]
            token_type_ids = token_type_ids[:, : self.max_text_len]

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        inputs["token_type_ids"] = token_type_ids
        inputs["position_ids"] = position_ids
        inputs["text_token_mask"] = text_self_attention_masks
        # end_time = time.perf_counter()  # 高精度结束计时
        # elapsed_ms = (end_time - start_time) * 1000  # 转换为毫秒
        # print(f"文本预处理耗时: {elapsed_ms:.3f} 毫秒")
        outputs = self.net.run(self.output_names, inputs)

        prediction_logits_ = np.squeeze(outputs[0], axis=0)  # [0]  # prediction_logits.shape = (nq, 256)
        prediction_logits_ = 1 / (1 + np.exp(-prediction_logits_))

        prediction_boxes_ = np.squeeze(outputs[1], axis=0)  # [0]  # prediction_boxes.shape = (nq, 4)

        filt_mask = np.max(prediction_logits_, axis=1) > infer_box_threshold
        logits_filt = prediction_logits_[filt_mask]  # num_filt, 256
        boxes_filt = prediction_boxes_[filt_mask]  # num_filt, 4

        pred_phrases = []
        valid_boxes = []
        valid_scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, input_ids[0, :], self.tokenizer)
            if pred_phrase is None:
                continue
            pred_phrases.append(pred_phrase)
            valid_boxes.append(box)
            valid_scores.append(float(np.max(logit)))

        if not valid_boxes:
            return np.empty((0, 4), dtype=prediction_boxes_.dtype), []

        boxes_filt = np.asarray(valid_boxes)
        scores = np.asarray(valid_scores)
        xyxy = boxes_filt.copy()
        xyxy[:, :2] -= xyxy[:, 2:] * 0.5
        xyxy[:, 2:] += xyxy[:, :2]
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            current = order[0]
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            xx1 = np.maximum(xyxy[current, 0], xyxy[remaining, 0])
            yy1 = np.maximum(xyxy[current, 1], xyxy[remaining, 1])
            xx2 = np.minimum(xyxy[current, 2], xyxy[remaining, 2])
            yy2 = np.minimum(xyxy[current, 3], xyxy[remaining, 3])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            current_area = max(0.0, xyxy[current, 2] - xyxy[current, 0]) * max(
                0.0, xyxy[current, 3] - xyxy[current, 1]
            )
            remaining_area = np.maximum(0.0, xyxy[remaining, 2] - xyxy[remaining, 0]) * np.maximum(
                0.0, xyxy[remaining, 3] - xyxy[remaining, 1]
            )
            union = current_area + remaining_area - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[iou <= nms_threshold]
        boxes_filt = boxes_filt[keep]
        pred_phrases = [pred_phrases[i] for i in keep]
        # end_time1 = time.perf_counter()  # 高精度结束计时
        # elapsed_ms1 = (end_time1 - end_time) * 1000  # 转换为毫秒
        # print(f"推理处理耗时: {elapsed_ms1:.3f} 毫秒")
        return boxes_filt, pred_phrases


def draw_boxes_to_image(image, boxes, labels):
    h, w = image.shape[:2]
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * np.array([w, h, w, h])
        # from xywh to xyxy
        box[:2] -= box[2:] * 0.5
        box[2:] += box[:2]

        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        # txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        # cv2.rectangle(image, (xmin, ymin + 1), (xmin + txt_size[0] + 1, ymin + int(1.5 * txt_size[1])), (255, 255, 255), -1)
        cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="D:/code/labelme_sam/labelme/ai/seg_model/GroundingDINO_SwinT_OGC.onnx",
                        help="onnx model path")
    parser.add_argument("--image_path", type=str, default=r"C:\Users\fjl\Desktop\240\raw_data\640_640_uav/01002050.jpg",
                        help="path to image file")
    parser.add_argument("--text_prompt", type=str, default="drone . ",
                        help="text prompt, 每个类别名称之间以 . 隔开")  ###cat_dog.jpeg的提示词:"chair . person . dog .""     demo7.jpg的提示词:"Horse . Clouds . Grasses . Sky . Hill ."
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    args = parser.parse_args()

    mynet = GroundingDINO(args.model_path, args.box_threshold, "D:\code\labelme_sam\labelme/ai\seg_model/vocab.txt", "cpu",text_threshold=args.text_threshold)
    srcimg = cv2.imread(args.image_path)

    boxes_filt, pred_phrases = mynet.detect(srcimg, args.text_prompt)
    drawimg = draw_boxes_to_image(srcimg, boxes_filt, pred_phrases)

    cv2.imwrite('result.jpg', drawimg)
    winName = 'GroundingDINO use OnnxRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
