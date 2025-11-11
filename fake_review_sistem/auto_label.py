# === auto_label.py ===
from textblob import TextBlob

def auto_label(text: str) -> int:
    """
    Sistem otomatis menentukan apakah ulasan Asli (1) atau Palsu (0)
    Berdasarkan sentimen dan kata-kata kunci.
    """
    if not text:
        return 0

    text = text.lower()
    sentiment = TextBlob(text).sentiment.polarity

    promo_words = ["murah", "diskon", "recommended", "puas banget", "cepat banget", "terbaik", "mantap"]
    fake_patterns = ["produk bagus", "sesuai deskripsi", "mantap sekali"]  # sering muncul di ulasan spam
    neg_words = ["jelek", "cacat", "rusak", "mengecewakan", "tidak sesuai"]

    # heuristik otomatis:
    if any(word in text for word in fake_patterns):
        return 0  # kemungkinan palsu (spam template)
    elif any(word in text for word in promo_words) and sentiment > 0.4:
        return 0  # kemungkinan palsu (terlalu promosi)
    elif any(word in text for word in neg_words):
        return 1  # kemungkinan asli (pengalaman jujur)
    elif sentiment > 0.3:
        return 1  # asli natural
    else:
        return 0  # netral / tidak jelas dianggap palsu