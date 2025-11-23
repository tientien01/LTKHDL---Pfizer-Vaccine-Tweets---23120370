import re
from collections import Counter
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import *

# ================================================================================================
# HASHTAG 
# ================================================================================================
def count_hashtags(text):
    """Sử dụng Regex để đếm số lượng Hashtag (#) trong một chuỗi."""
    # Pattern: # theo sau là một hoặc nhiều ký tự chữ/số/gạch dưới (\w+)
    try:
        # Đảm bảo xử lý đúng các kiểu string từ numpy
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        return len(re.findall(r'#\w+', text))
    except TypeError:
        return 0 
def process_hashtag(hashtag_string):
    if isinstance(hashtag_string, bytes):
        hashtag_string = hashtag_string.decode('utf-8')

    # chuyển chữ thường
    hashtag_string = hashtag_string.lower()

    # Loại bỏ các dấu
    hashtag_string = re.sub(r'[^\w\s]', '', hashtag_string)

    # Bỏ các khoảng trắng
    hashtag_string = re.sub(r'\s+', ' ', hashtag_string).strip()

    return hashtag_string

# ================================================================================================
# MENTION
# ================================================================================================
def count_mentions(text):
    """Sử dụng Regex để đếm số lượng Mentions (@) trong một chuỗi."""
    # Pattern: @ theo sau là một hoặc nhiều ký tự chữ/số/gạch dưới (\w+)
    try:
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        return len(re.findall(r'@\w+', text))
    except TypeError:
        return 0 
    

# ================================================================================================
# TEXT PROCESS
# ================================================================================================
my_stop_words = set(ENGLISH_STOP_WORDS).union({
    'https', 'http', 'com', 'www', 'twitter', 'pic', 'status', # Rác link
    'pfizer', 'vaccine', 'covid', 'covid19', 'biontech' # Từ khóa chủ đề (xuất hiện quá nhiều nên lọc bỏ để thấy cái khác)
})

def get_keywords_simple(text_arr, top_k=10):
    all_words = []
    for t in text_arr:
        # 1. Xử lý byte string nếu cần
        if isinstance(t, bytes):
            t = t.decode('utf-8')
        else:
            t = str(t)
            
        # 2. Tách từ đơn giản bằng Regex (chỉ lấy chữ cái/số)
        # \w+ lấy các ký tự chữ và số, bỏ qua dấu câu
        w_list = re.findall(r'\w+', t.lower())
        
        # 3. Lọc từ: Không nằm trong stop words VÀ độ dài > 3
        filtered_words = [w for w in w_list if w not in my_stop_words and len(w) > 3]
        
        all_words.extend(filtered_words)
    
    # 4. Đếm và trả về top K
    return Counter(all_words).most_common(top_k)

def clean_text_advanced(text):
    """Làm sạch + Xử lý Emoji + Stemming"""
    if not isinstance(text, str): return ""
    for emo, meaning in EMOJI_MAP.items():
        if emo in text: text = text.replace(emo, f" {meaning} ")
    text = text.lower()
    text = text.replace("n't", " not") 
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    stemmed_words = [simple_stemmer(w) for w in words]
    return " ".join(stemmed_words)

def generate_sentiment_label_advanced(text):
    """Gán nhãn (Silver Labels)"""
    words = text.split()
    score = 0
    for i, word in enumerate(words):
        is_negated = False
        if i > 0 and words[i-1] in NEGATION_WORDS:
            is_negated = True
        val = 0
        if word in POSITIVE_WORDS: val = 1
        elif word in NEGATIVE_WORDS: val = -1
        if is_negated: val = val * -1
        score += val
    if score > 0: return 2
    elif score < 0: return 0
    else: return 1


def is_basic_vowel(ch: str) -> bool:
    """Kiểm tra nguyên âm cơ bản (a, e, i, o, u)"""
    return ch in "aeiou"

# --- Các Hàm Phụ Trợ Bổ Sung cho Porter Stemmer ---
def is_vowel_porter(word: str, i: int) -> bool:
    """
    Xác định nguyên âm theo quy tắc Porter, bao gồm xử lý chữ 'y'.
    'y' được coi là nguyên âm nếu nó theo sau một phụ âm và không đứng đầu từ.
    """
    ch = word[i].lower()
    if is_basic_vowel(ch):
        return True
    if ch == 'y':
        # 'y' là nguyên âm nếu nó không phải ký tự đầu tiên 
        # và ký tự trước đó là phụ âm.
        if i > 0 and not is_basic_vowel(word[i-1]):
            return True
    return False

def get_v_c_pattern(word: str) -> str:
    """Tạo chuỗi ký hiệu V/C cho từ dựa trên quy tắc Porter."""
    pattern = ""
    for i in range(len(word)):
        if is_vowel_porter(word, i):
            pattern += "V"
        else:
            pattern += "C"
    return pattern

def calculate_m(word: str) -> int:
    """
    Tính độ đo m (measure) của từ. 
    m là số lần lặp lại của mẫu (VC), ví dụ: C(VC)^m V
    """
    if not word:
        return 0
    
    # Lấy chuỗi pattern C/V
    vc_pattern = get_v_c_pattern(word)
    
    # Đếm số lần chuyển đổi từ C -> V -> C
    # Mẫu cần tìm: C V C V C V...
    
    m = 0
    # Bắt đầu tìm kiếm từ V đầu tiên
    start_index = -1
    for i, char in enumerate(vc_pattern):
        if char == 'V':
            start_index = i
            break
            
    if start_index == -1: # Không có nguyên âm nào
        return 0
        
    i = start_index
    while i < len(vc_pattern):
        # Bước 1: Quét qua tất cả các V liên tiếp (đầu tiên)
        if vc_pattern[i] == 'V':
            while i < len(vc_pattern) and vc_pattern[i] == 'V':
                i += 1
            
            # Bước 2: Quét qua tất cả các C liên tiếp (thứ hai)
            if i < len(vc_pattern) and vc_pattern[i] == 'C':
                while i < len(vc_pattern) and vc_pattern[i] == 'C':
                    i += 1
                
                # Nếu chuỗi tiếp tục với V, ta đã tìm thấy một cặp (VC)
                if i < len(vc_pattern) and vc_pattern[i] == 'V':
                    m += 1
                else:
                    break # Kết thúc bằng C hoặc hết chuỗi
            else:
                break # Kết thúc bằng V hoặc hết chuỗi
        else:
            # Bắt đầu không phải V (tức là C)
            i += 1 
            
    return m

# --- Hàm Cắt Gốc Gốc ---
def contains_vowel(word: str) -> bool:
    """Kiểm tra xem từ có chứa ít nhất một nguyên âm theo quy tắc Porter không."""
    for i in range(len(word)):
        if is_vowel_porter(word, i):
            return True
    return False

def ends_with_double_consonant(word: str) -> bool:
    """Kiểm tra phụ âm đôi ở cuối (trừ l, s, z)"""
    if len(word) < 2:
        return False
    # Phải là hai chữ cái giống nhau và không phải nguyên âm theo Porter
    return (word[-1] == word[-2] and 
            not is_vowel_porter(word, len(word) - 1))

def cvc(word: str) -> bool:
    """Kiểm tra mẫu C - V - C cuối, với C2 không phải w, x, y."""
    if len(word) < 3:
        return False
    
    c1, v, c2 = word[-3], word[-2], word[-1]
    
    # Kiểm tra C-V-C theo Porter
    return (not is_vowel_porter(word, len(word)-3)) and \
           is_vowel_porter(word, len(word)-2) and \
           (not is_vowel_porter(word, len(word)-1)) and \
           c2 not in "wxy"

def simple_stemmer(word: str) -> str:
    word = word.lower()

    if len(word) <= 2:
        return word

    # Bắt đầu với việc xử lý chữ 'y' (Step 1c) 
    # Nếu 'y' theo sau phụ âm, chuyển y thành i để các bước sau xử lý tốt hơn.
    if word.endswith("y") and len(word) > 1 and not is_basic_vowel(word[-2]):
        word = word[:-1] + "i"
        
    # --- Step 1a: Xử lý dạng số nhiều và các đuôi phổ biến ---
    # THỨ TỰ QUAN TRỌNG
    if word.endswith("sses"):
        word = word[:-2] # caresses -> caress
    elif word.endswith("ies"):
        word = word[:-2] # ponies -> poni
    elif word.endswith("ss"):
        pass # giữ nguyên 'ss' (ví dụ: stress -> stress)
    elif word.endswith("s"):
        word = word[:-1] # cats -> cat

    # --- Step 1b: Xử lý 'ed', 'ing' ---
    # THỨ TỰ QUAN TRỌNG
    
    # 1. 'eed' (yêu cầu m > 0)
    if word.endswith("eed"):
        base = word[:-3]
        m = calculate_m(base)
        if m > 0:
            word = word[:-1] # agreed -> agree
            
    # 2. 'ed' (yêu cầu base chứa nguyên âm)
    elif word.endswith("ed"):
        base = word[:-2]
        if contains_vowel(base):
            word = base
            
            # --- Quy tắc bổ sung sau khi loại bỏ 'ed' ---
            # a) Nếu kết thúc bằng phụ âm đôi (trừ l, s, z), loại bỏ một
            if ends_with_double_consonant(word) and word[-1] not in "lsz":
                 word = word[:-1]
            # b) Nếu kết thúc bằng CVC (c2 không phải w, x, y) và m=1, thêm 'e'
            elif cvc(word) and calculate_m(word) == 1:
                 word += "e"
            # c) Nếu kết thúc bằng 'at', 'bl', 'iz', thêm 'e'
            elif word.endswith(("at", "bl", "iz")):
                word += "e"
            
    # 3. 'ing' (yêu cầu base chứa nguyên âm)
    elif word.endswith("ing"):
        base = word[:-3]
        if contains_vowel(base):
            word = base
            
            # --- Quy tắc bổ sung sau khi loại bỏ 'ing' ---
            # a) Nếu kết thúc bằng phụ âm đôi (trừ l, s, z), loại bỏ một
            if ends_with_double_consonant(word) and word[-1] not in "lsz":
                 word = word[:-1]
            # b) Nếu kết thúc bằng CVC (c2 không phải w, x, y) và m=1, thêm 'e'
            elif cvc(word) and calculate_m(word) == 1:
                 word += "e"
            # c) Nếu kết thúc bằng 'at', 'bl', 'iz', thêm 'e'
            elif word.endswith(("at", "bl", "iz")):
                word += "e"

    # --- Step 2: Xử lý hậu tố phức tạp (Yêu cầu m > 0) ---
    m_current = calculate_m(word)
    
    suffix_map = {
        "ational": "ate",
        "tional": "tion",
        "izer": "ize",
        "alism": "al",
        "aliti": "al",
        "fullness": "ful",
        "ousness": "ous",
        "iveness": "ive",
        "biliti": "ble",
        "enci": "ence",
        "anci": "ance",
        "logi": "log"
    }

    for suf, rep in suffix_map.items():
        if word.endswith(suf):
            if calculate_m(word[: -len(suf)]) > 0:
                word = word[: -len(suf)] + rep
                break
                
    # còn thiếu, nhưng code này đã đủ để minh họa cách dùng m.
    
    return word



# ================================================================================================
# LOCATION 
# ================================================================================================
# Hàm Substring Match
def improved_location_mapper(normalized_location, mapping_dict):
    loc_lower = normalized_location.strip()
    
    for key_part, standardized_region in mapping_dict.items():
        # Chỉ khớp các từ khóa có độ dài > 1
        if len(key_part) > 1 and key_part in loc_lower:
            return standardized_region 
    
    # Nếu không khớp, giữ nguyên chuỗi đã được làm sạch
    return loc_lower
