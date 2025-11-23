
import numpy as np

# ================================
# LOCATION 
# ================================
# Dictionary ánh xạ phức tạp
COMPREHENSIVE_MAPPING = {
    # Ưu tiên các từ viết tắt và khu vực lớn nhất để nhóm về USA
    'usa': 'usa', 'united states': 'usa', 'california': 'usa', 'new york': 'usa', 'texas': 'usa', 
    'ny': 'usa', 'tx': 'usa', 'fl': 'usa', 'pa': 'usa', 'nj': 'usa', 'ca': 'usa', 'dc': 'usa', 
    'ga': 'usa', 'il': 'usa', 'wa': 'usa', 'nc': 'usa', 'oh': 'usa', 'mo': 'usa', 'az': 'usa',
    'los angeles': 'usa',
    
    # Vương quốc Anh/Châu Âu
    'uk': 'united kingdom', 'england': 'united kingdom', 'london': 'united kingdom', 
    'scotland': 'united kingdom', 'wales': 'united kingdom', 'ireland': 'ireland',
    'germany': 'germany', 'france': 'france', 'europe': 'europe',  'glasgow environs': 'united kingdom',  
    
    # Châu Á/UAE/Canada
    'india': 'india', 'mumbai': 'india', 'delhi': 'india', 'kolkata': 'india', 'chennai': 'india',
    'uae': 'united arab emirates', 'dubai': 'united arab emirates', 'abu dhabi': 'united arab emirates',
    'canada': 'canada', 'toronto': 'canada', 'ontario': 'canada', 'montreal': 'canada',
    'malaysia': 'malaysia', 'petaling jaya': 'malaysia', 'singapore': 'singapore', 'hong kong': 'hong kong',
    
    # Global/Others
    'global': 'global', 'earth': 'global', 'worldwide': 'global', 'n/a': "unknown_location",
    'unknown': "unknown_location" 
}


# ================================
# TEXT
# ================================

# 1. Cấu hình Emoji & Từ điển (Dữ liệu nền tảng)
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'safe', 
    'effective', 'thanks', 'thankful', 'grateful', 'hope', 'success', 'happy', 
    'protection', 'relief', 'excited', 'glad', 'perfect', 'awesome', 'better',


    # 1. Nhóm Khen ngợi mức độ cao (Superlatives)
    'fantastic', 'incredible', 'brilliant', 'outstanding', 'superb', 
    'magnificent', 'legendary', 'impressive', 'top', 'class', 'phenomenal',
    
    # 2. Nhóm Niềm tin & An tâm (Rất quan trọng trong y tế/dịch vụ)
    'trust', 'trusted', 'confident', 'confidence', 'secure', 'reassured', 
    'reliable', 'trustworthy', 'calm', 'relax', 'relaxed', 'comfort', 'comfortable',
    
    # 3. Nhóm Trải nghiệm suôn sẻ (Ví dụ: tiêm không đau, thủ tục nhanh)
    'easy', 'easier', 'easiest', 'simple', 'smooth', 'smoothly', 
    'quick', 'fast', 'painless', 'gentle', 'organized', 'efficient',
    
    # 4. Nhóm Lợi ích & Giải pháp
    'beneficial', 'benefit', 'helpful', 'valuable', 'advantage', 
    'solution', 'cure', 'remedy', 'improvement', 'improved', 'protect',
    
    # 5. Nhóm Chiến thắng & Ủng hộ
    'win', 'winning', 'victory', 'triumph', 'achievement', 'accomplished',
    'support', 'endorse', 'recommend', 'recommended', 'encourage',
    
    # 6. Nhóm Từ cảm thán/Slang (Thường gặp trên Twitter/MXH)
    'yay', 'hurray', 'woohoo', 'bravo', 'kudos', 'cheers', 
    'cool', 'nice', 'lovely', 'pleasant', 'enjoy', 'enjoyed'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dangerous', 'risk',
    'fear', 'scared', 'fail', 'failed', 'death', 'sick', 'pain', 'hurt', 'harm', 
    'useless', 'fake', 'scam', 'problem', 'severe', 'worry', 'sad',
    # 1. Nhóm Tác dụng phụ & Đau đớn (Rất quan trọng trong y tế)
    'sore', 'soreness', 'ache', 'aching', 'fever', 'chills', 'nausea', 'vomit',
    'fatigue', 'tired', 'exhausted', 'dizzy', 'headache', 'migraine', 'swollen',
    'rash', 'itchy', 'bruise', 'weak', 'weakness', 'ill', 'illness', 'suffer', 'suffering',
    
    # 2. Nhóm Sợ hãi & Lo lắng
    'anxious', 'anxiety', 'panic', 'terrified', 'nervous', 'worrying', 'concerned',
    'afraid', 'dread', 'dreading', 'uneasy', 'stress', 'stressed', 'horror',
    
    # 3. Nhóm Nghi ngờ & Mất niềm tin (Quan trọng để lọc tin giả/anti-vax)
    'lie', 'lying', 'liar', 'suspicious', 'propaganda', 'conspiracy', 'untrusted',
    'skeptical', 'doubt', 'doubtful', 'misleading', 'false', 'hoax', 'cheat',
    'unsafe', 'risky', 'threat', 'poison', 'toxic',
    
    # 4. Nhóm Thất vọng & Chê bai
    'disappointed', 'disappointing', 'useless', 'waste', 'pointless', 'stupid',
    'incompetent', 'mess', 'chaos', 'shame', 'disgrace', 'pathetic', 'poor',
    'slow', 'delayed', 'late', 'refused', 'rejected', 'denied',
    
    # 5. Nhóm Giận dữ & Phản đối
    'angry', 'furious', 'annoyed', 'annoying', 'mad', 'upset', 'frustrated',
    'complain', 'complaint', 'hell', 'damn', 'wtf', 'ridiculous', 'crazy'
}

EMOJI_MAP = {
    # Tích cực (Positive)
    "😀": "happy", "😃": "happy", "😄": "happy", "😁": "happy",
    "😊": "happy", "😍": "love",  "😘": "love",  "🥰": "love",
    "😂": "funny", "🤣": "funny", "😅": "funny",
    "👍": "good",  "👏": "clap",  "🙏": "thank", "💪": "strong",
    "❤️": "love",  "🧡": "love",  "💛": "love",  "💚": "love", "💙": "love",
    "🎉": "celebrate", "✨": "shiny", "💯": "perfect", "🙌": "support",
    
    # Tiêu cực (Negative)
    "😢": "sad",   "😭": "sad",   "😞": "sad",   "😔": "sad",
    "😡": "angry", "😠": "angry", "🤬": "angry", "😤": "angry",
    "👎": "bad",   "💔": "heartbreak",
    "🤮": "disgust", "🤢": "disgust",
    "😱": "scared",  "😨": "scared",  "wv": "scared",
    "🤯": "shock",   "😳": "shock",   "🙄": "annoyed",
    "🤦": "facepalm", "😑": "bored", "😒": "annoyed",

    # Y tế / Vaccine (Rất quan trọng cho bài toán của bạn)
    "💉": "vaccine", 
    "🦠": "virus", 
    "😷": "mask", 
    "🤒": "sick", 
    "🤕": "pain", 
    "🚑": "ambulance", 
    "🏥": "hospital", 
    "💊": "medicine",
    "🩺": "doctor",
    "☠️": "death",
    
    # Khác
    "📢": "announce", "🚨": "alert", "🤔": "thinking"
}
NEGATION_WORDS = {
    # Phủ định cơ bản
    'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'nowhere',
    
    # Các dạng viết tắt của "n't" (có và không có dấu ')
    "n't", 'cannot', 
    'cant', "can't",
    'dont', "don't",
    'wont', "won't",
    'isnt', "isn't",
    'arent', "aren't",
    'aint', "ain't",
    'wasnt', "wasn't",
    'werent', "weren't",
    'hasnt', "hasn't",
    'havent', "haven't",
    'hadnt', "hadn't",
    'doesnt', "doesn't",
    'didnt', "didn't",
    'couldnt', "couldn't",
    'shouldnt', "shouldn't",
    'wouldnt', "wouldn't",
    'mustnt', "mustn't",
    
    # Từ mang nghĩa phủ định ngữ cảnh (Contextual negations)
    'without', 'lack', 'missing',
    'barely', 'hardly', 'scarcely', 'rarely'
}

# Nguyên âm dùng để kiểm tra
VOWELS = np.array(list("aeiou"))
