import json
import ctypes
import numpy as np
from blingfire import *
from numpy.ctypeslib import ndpointer

blingfire_path = "./blingfiretokdll.dll"
blingfire_model = "./data/xlm_roberta_base.bin"
vocab_path = "./data/vocab.txt"
max_doc_count = 96
max_seq_length = 256
max_query_length = 16
max_title_length = 32
max_url_length = 32

h = load_model(blingfire_model)

print("Load Bling Fire Tokenizer")

dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["PATH"] = dir_path + ';' + os.environ["PATH"]
ranklm_lib = ctypes.CDLL("./RankLMTokenization.dll")

ranklm_init = ranklm_lib.RankLMTokenization_SentencePiece_FBV_Init
ranklm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
ranklm_init.restype = None

ranklm_id_tokenize = ranklm_lib.RankLMTokenization_SentencePiece_FBV_ID_Tokenize
ranklm_id_tokenize.argtypes = [ctypes.c_char_p, ndpointer(ctypes.c_int), 
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int), 
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_int, ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32)]
ranklm_id_tokenize.restype = None

ranklm_token_tokenize = ranklm_lib.RankLMTokenization_SentencePiece_FBV_Token_Tokenize
ranklm_token_tokenize.argtypes = [ctypes.c_char_p, ndpointer(ctypes.c_int), 
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int), 
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_int, ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32)]
ranklm_token_tokenize.restype = None

ranklm_tokenize = ranklm_lib.RankLMTokenization_SentencePiece_FBV_Tokenize
ranklm_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ndpointer(ctypes.c_int), 
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int), 
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_int, ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32)]
ranklm_tokenize.restype = None

ranklm_fb_tokenize = ranklm_lib.RankLMTokenization_SentencePiece_FBV_FB_Tokenize
ranklm_fb_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ndpointer(ctypes.c_int), 
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int), 
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_char_p, ndpointer(ctypes.c_int),
                                ctypes.c_int, ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32)]
ranklm_fb_tokenize.restype = None

ranklm_init(max_doc_count, max_seq_length, max_query_length, max_title_length, max_url_length, ctypes.c_char_p(blingfire_path.encode("utf-8")), ctypes.c_char_p(vocab_path.encode("utf-8")))

def get_lang_dist_from_market(market):
    lang_dist = market.split('-')
    if len(lang_dist) >= 2:
        language = "-".join(lang_dist[:-1])
        district = lang_dist[-1]
    else:
        language = "un"
        district = "un"

    return language, district

def get_lang_dist(market, market_json):

    if ("Language" in market_json) and ("Region" in market_json):
        lang_dist_dict = json.loads(market_json)
        language = lang_dist_dict["Language"].lower().strip()
        district = lang_dist_dict["Region"].lower().strip()

        if language == "" or district == "":
            language, district = get_lang_dist_from_market(market)

    else:
        language, district = get_lang_dist_from_market(market)

    return language, district


input_list = [["23314 454 7560 85 5 3958 32 188131 454 11627 1369", "153 115 13761 3245 30128 21393 6 3958 6 33957 2011 126707 13820 18 75813 121046 6957 1284 18 46667 225006 153 24 33416 6 78175 111202 20179 95 39884 13639 425 16684 23314 194602 78403 2011 124999 153 196423 31 9607 363 36398 96335 68828 9351 45 10763 6635 7026 8834 73395 1230 82678 74", "106 25037 92 6 2566 3114 64 9271 41793 92", "48498 100 71 77463 26249 36049 141496 159201 41 1294 22970 144", "fr-fr", ""], ["11493 5 337 67847", "305 13312 6650 20 351 1507 1202 337 67847 337 67847 11493 123 3177", "78600 30535 113543 81384 64 10248 64 864 910 2507 169 3742 6 7693", "337 67847 11493 123 3177 20 337 67847 35399", "en-id", ""], ["6 8709 71684 1128 56963 9594", "378 122369 268 6 8709 71684 1128 4035 9056 11541 64632 37106 46879 2490 9839 5873 5 1210 37151 153 28292 194546 56963 18617 143964 9594 15 6 192141 10134 2846 1388 6 167039 8709 71684 1128 106000 194546 240762 6995 1173 35645 684 109052 5873 15 6 20212 10134 2846 1388 6 71729 38", "82414 496 9365 65451", "6 8709 71684 1128 14455 9065 9 12865 68818 1764", "zh-tw", ""]]

query_list = b""
snippet_list = b""
url_list = b""
title_list = b""
lang_list = b""
dist_list = b""

query_lengths = []
snippet_lengths = []
url_lengths = []
title_lengths = []
lang_lengths = []
dist_lengths = []

for instance in input_list:
    
    query = instance[0].strip()
    snippet = instance[1].strip() + " " + instance[5].strip()
    url = instance[2].strip()
    title = instance[3].strip()
    market = instance[4].lower()
    language, district = get_lang_dist(market, instance[-1])

    query_encode = query.encode("utf-8")
    snippet_encode = snippet.encode("utf-8")
    url_encode = url.encode("utf-8")
    title_encode = title.encode("utf-8")
    lang_encode = language.encode("utf-8")
    dist_encode = district.encode("utf-8")

    query_list += query_encode
    snippet_list += snippet_encode
    url_list += url_encode
    title_list += title_encode
    lang_list += lang_encode
    dist_list += dist_encode

    query_lengths.append(len(query_encode))
    snippet_lengths.append(len(snippet_encode))
    url_lengths.append(len(url_encode))
    title_lengths.append(len(title_encode))
    lang_lengths.append(len(lang_encode))
    dist_lengths.append(len(dist_encode))

p_query_list = ctypes.c_char_p(query_list)
p_snippet_list = ctypes.c_char_p(snippet_list)
p_url_list = ctypes.c_char_p(url_list)
p_title_list = ctypes.c_char_p(title_list)
p_lang_list = ctypes.c_char_p(lang_list)
p_dist_list = ctypes.c_char_p(dist_list)

p_query_lengths = np.array(query_lengths, dtype="int32")
p_snippet_lengths = np.array(snippet_lengths, dtype="int32")
p_url_lengths = np.array(url_lengths, dtype="int32")
p_title_lengths = np.array(title_lengths, dtype="int32")
p_lang_lengths = np.array(lang_lengths, dtype="int32")
p_dist_lengths = np.array(dist_lengths, dtype="int32")

batch_size = len(query_lengths)

input_ids = np.zeros((batch_size, max_seq_length), dtype="int32")
segment_ids = np.zeros((batch_size, max_seq_length), dtype="int32")
input_mask = np.zeros((batch_size, max_seq_length), dtype="int32")

ranklm_id_tokenize(p_query_list, p_query_lengths,
            p_snippet_list, p_snippet_lengths,
            p_url_list, p_url_lengths,
            p_title_list, p_title_lengths,
            p_lang_list, p_lang_lengths,
            p_dist_list, p_dist_lengths,
            batch_size, input_ids, segment_ids, input_mask)


print("input_ids: ", input_ids)
print("segment_ids: ", segment_ids)
print("input_mask: ", input_mask)

free_model(h)