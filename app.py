import os
import re
from collections import deque
from functools import lru_cache
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import INTENTS, DOCUMENTS

# 1. Memuat Environment & Konfigurasi
load_dotenv(override=True)
T_HIGH = float(os.getenv("T_HIGH", "0.65"))
T_LOW = float(os.getenv("T_LOW", "0.10"))
TOP_K = int(os.getenv("TOP_K", "3"))

app = Flask(__name__)


def _get_gemini_client():
    load_dotenv(override=True)
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

SENTINEL_TIDAK_TAHU = "TIDAK_TAHU"
PESAN_TIDAK_TAHU = (
    "Mohon maaf, saya tidak tahu. Informasi tersebut belum tersedia "
    "di dalam basis pengetahuan akademik UNP."
)

# Memori percakapan singkat di RAM (tanpa database) untuk mendukung pertanyaan lanjutan
riwayat_percakapan = deque(maxlen=4)

# Daftar stopword & normalisasi sinonim Bahasa Indonesia agar TF-IDF tahan variasi kata
STOPWORDS_ID = {
    "yang", "di", "ke", "dari", "pada", "dalam", "untuk", "dengan", "dan", "atau",
    "ini", "itu", "adalah", "sebagai", "oleh", "saya", "aku", "anda", "kami", "kita",
    "apa", "siapa", "dimana", "kapan", "bagaimana", "berapa", "mengapa", "kenapa",
    "apakah", "bisakah", "tolong", "mohon", "mau", "ingin", "cara", "lihat", "cek",
    "hari", "saat", "sekarang", "ada", "mana", "ya", "dong", "nih", "sih", "nya",
}

SINONIM_MAP = {
    "uang semester": "ukt",
    "uang semesteran": "ukt",
    "biaya semester": "ukt",
    "biaya kuliah": "ukt",
    "uang kuliah": "ukt",
    "spp": "ukt",
    "berhenti studi": "cuti akademik",
    "bss": "cuti akademik",
    "ambil sks": "pengisian krs",
    "isi krs": "pengisian krs",
    "kartu rencana studi": "krs",
    "maba": "mahasiswa baru",
    "pmb": "penerimaan mahasiswa baru",
    "perpustakaan": "pustaka perpustakaan",
    "perpus": "pustaka perpustakaan",
    "pimpinan": "rektor",
    "ketua kampus": "rektor rektor unp",
    "kepala kampus": "rektor rektor unp",
    "letak": "lokasi alamat",
    "posisi": "lokasi alamat",
    "lupa password": "lupa kata sandi kendala login",
    "tidak bisa login": "kendala login siakad",
}

KATA_RUJUKAN = {"nya", "beliau", "tersebut", "itu", "disana", "situ", "kantornya", "syaratnya", "caranya"}


# 2. Preprocessing Teks (Normalisasi Sinonim + Pembersihan Tanda Baca)
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Ekspansi sinonim frasa agar pencocokan kata lebih akurat
    for frasa, padanan in SINONIM_MAP.items():
        if frasa in text:
            text = f"{text} {padanan}"

    tokens = [t for t in text.split() if t not in STOPWORDS_ID]
    return " ".join(tokens) if tokens else text


# 3. Persiapan Hybrid TF-IDF Vectorizer (Word N-Gram + Char N-Gram agar tahan Typo)
intent_corpus = []
intent_labels = []
response_map = {}

for intent_name, data in INTENTS.items():
    response_map[intent_name] = data["response"]
    for ex in data["examples"]:
        intent_corpus.append(preprocess(ex))
        intent_labels.append(intent_name)

# Kombinasi Word TF-IDF dan Char-WB TF-IDF (toleran salah ketik / typo ringan)
intent_word_vec = TfidfVectorizer(ngram_range=(1, 2))
intent_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
intent_word_matrix = intent_word_vec.fit_transform(intent_corpus)
intent_char_matrix = intent_char_vec.fit_transform(intent_corpus)

doc_corpus = [
    preprocess(f"{doc['judul']} {doc['isi']}") for doc in DOCUMENTS
]
doc_word_vec = TfidfVectorizer(ngram_range=(1, 2))
doc_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
doc_word_matrix = doc_word_vec.fit_transform(doc_corpus)
doc_char_matrix = doc_char_vec.fit_transform(doc_corpus)


def _hitung_kemiripan_gabungan(teks_terproses: str, word_vec, char_vec, word_mat, char_mat):
    """Menggabungkan skor kemiripan kata (75%) dan karakter n-gram (25%) untuk ketahanan typo."""
    w_q = word_vec.transform([teks_terproses])
    c_q = char_vec.transform([teks_terproses])
    sim_word = cosine_similarity(w_q, word_mat)[0]
    sim_char = cosine_similarity(c_q, char_mat)[0]
    return (0.75 * sim_word) + (0.25 * sim_char)


# 4. Pencocokan Intent Rule-Based (sebagai pembanding evaluasi Modul 5)
def match_intent_rule_based(user_input: str):
    raw_clean = re.sub(r"[^\w\s]", " ", user_input.lower()).strip()
    raw_clean = re.sub(r"\s+", " ", raw_clean)
    for intent_name, data in INTENTS.items():
        for pattern in data["examples"]:
            pat_clean = re.sub(r"[^\w\s]", " ", pattern.lower()).strip()
            if pat_clean in raw_clean:
                return intent_name
    return None


# 5. Pencocokan Intent TF-IDF dan Pengambilan Top-K Dokumen
def match_intent_tfidf(user_input: str):
    processed = preprocess(user_input)
    if not processed:
        return None, None, 0.0

    similarities = _hitung_kemiripan_gabungan(
        processed, intent_word_vec, intent_char_vec, intent_word_matrix, intent_char_matrix
    )
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    matched_intent = intent_labels[best_idx]
    return matched_intent, response_map[matched_intent], best_score


def cari_top_k(pesan: str, k: int = TOP_K):
    """Mengembalikan (skor_terbaik, daftar_dokumen_top_k) berdasarkan TF-IDF."""
    processed = preprocess(pesan)
    if not processed:
        return 0.0, []

    similarities = _hitung_kemiripan_gabungan(
        processed, doc_word_vec, doc_char_vec, doc_word_matrix, doc_char_matrix
    )

    urut_idx = similarities.argsort()[::-1][:k]
    skor_terbaik = float(similarities[urut_idx[0]]) if len(urut_idx) > 0 else 0.0

    top_docs = []
    for idx in urut_idx:
        if similarities[idx] > 0:
            top_docs.append({
                "id": DOCUMENTS[idx]["id"],
                "judul": DOCUMENTS[idx]["judul"],
                "isi": DOCUMENTS[idx]["isi"],
                "skor": float(similarities[idx]),
            })

    return skor_terbaik, top_docs


# 6. Pemanggilan Gemini dengan Prompt Terbatas Konteks + In-Memory LRU Cache
MODEL_CANDIDATES = [
    "gemini-4-hydrogen",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]


@lru_cache(maxsize=128)
def _panggil_gemini_cached(api_key: str, pesan_user: str, blok_konteks: str, teks_riwayat: str) -> str:
    client = genai.Client(api_key=api_key)
    system_prompt = (
        "Anda adalah asisten virtual resmi Universitas Negeri Padang (UNP).\n"
        "ATURAN KETAT:\n"
        "1. Jawab pertanyaan pengguna HANYA berdasarkan KONTEKS yang diberikan di bawah.\n"
        "2. Jika informasi di dalam KONTEKS tidak memuat jawaban atau tidak cukup untuk menjawab, "
        f"Anda WAJIB membalas persis dengan satu kata: {SENTINEL_TIDAK_TAHU}\n"
        "3. Dilarang keras menebak, berasumsi, atau menggunakan pengetahuan di luar KONTEKS.\n"
        "4. Jawab secara ringkas, sopan, dan jelas dalam Bahasa Indonesia."
    )

    bagian_riwayat = f"RIWAYAT PERCAKAPAN SEBELUMNYA:\n{teks_riwayat}\n\n" if teks_riwayat else ""
    user_prompt = f"{bagian_riwayat}KONTEKS:\n{blok_konteks}\n\nPERTANYAAN: {pesan_user}"

    last_err = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                ),
            )
            teks = (response.text or "").strip()
            if not teks or SENTINEL_TIDAK_TAHU in teks.upper():
                return SENTINEL_TIDAK_TAHU
            return teks
        except Exception as e:
            last_err = e
            continue
    raise last_err


def tanya_gemini(pesan_user: str, konteks_list: list) -> str:
    load_dotenv(override=True)
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key or not konteks_list:
        return SENTINEL_TIDAK_TAHU

    blok_konteks = "\n".join(
        [f"[{i + 1}] {d['judul']}: {d['isi']}" for i, d in enumerate(konteks_list)]
    )
    teks_riwayat = "\n".join(
        [f"User: {u} | Bot: {b}" for u, b in list(riwayat_percakapan)[-2:]]
    )

    try:
        return _panggil_gemini_cached(api_key, pesan_user, blok_konteks, teks_riwayat)
    except Exception as e:
        print(f"[ERROR GEMINI API]: {e}")
        return f"ERROR_API::{e}"


# 7. Router Hybrid 3 Jalur (KB Lokal -> RAG Terbatas Top-K -> Fallback)
def _perkaya_dengan_riwayat(pesan_user: str) -> str:
    """Jika pertanyaan pendek dan mengandung kata rujukan, gabungkan topik pertanyaan sebelumnya."""
    kata_input = set(re.sub(r"[^\w\s]", " ", pesan_user.lower()).split())
    ada_rujukan = bool(kata_input & KATA_RUJUKAN) or any(
        w.endswith("nya") and len(w) > 4 for w in kata_input
    )
    if ada_rujukan and len(riwayat_percakapan) > 0 and len(kata_input) <= 6:
        pesan_terakhir, _ = riwayat_percakapan[-1]
        return f"{pesan_terakhir} {pesan_user}"
    return pesan_user


def get_response_hybrid(pesan_user: str):
    kueri_pencarian = _perkaya_dengan_riwayat(pesan_user)

    # Hitung kemiripan pada intent (Jalur 1) dan dokumen (Jalur 2)
    intent, respons_lokal, skor_intent = match_intent_tfidf(kueri_pencarian)
    skor_doc, top_k_docs = cari_top_k(kueri_pencarian, k=TOP_K)

    skor_terbaik = max(skor_intent, skor_doc)

    # JALUR 1: Skor intent tinggi DAN lebih relevan daripada dokumen panjang -> jawab dari KB Lokal
    if skor_intent >= T_HIGH and skor_intent >= (skor_doc + 0.15):
        riwayat_percakapan.append((pesan_user, respons_lokal))
        return respons_lokal, "Knowledge Base Lokal (TF-IDF)", skor_intent

    # JALUR 2: Skor masuk rentang konteks -> ambil top-k dokumen & kirim ke Gemini
    if skor_terbaik >= T_LOW and len(top_k_docs) > 0:
        jawaban_llm = tanya_gemini(pesan_user, top_k_docs)

        if jawaban_llm == SENTINEL_TIDAK_TAHU:
            return PESAN_TIDAK_TAHU, "Fallback (Di Luar Cakupan)", skor_terbaik

        if jawaban_llm.startswith("ERROR_API"):
            riwayat_percakapan.append((pesan_user, PESAN_TIDAK_TAHU))
            detail = jawaban_llm.replace("ERROR_API::", "")
            if "BILLING_DISABLED" in detail:
                pesan_err = (
                    "Maaf, API Key 'AQ...' (Vertex AI Google Cloud) mewajibkan Billing/Kartu Kredit aktif. "
                    "Untuk tugas kuliah GRATIS tanpa kartu kredit, buat API Key di https://aistudio.google.com/apikey "
                    "(pilih 'Create API key in new project' agar mendapat key yang diawali 'AIzaSy...')."
                )
            elif "SERVICE_DISABLED" in detail:
                pesan_err = (
                    "Maaf, Vertex AI API belum diaktifkan pada Google Cloud Project Anda."
                )
            else:
                pesan_err = "Maaf, layanan AI sedang tidak dapat diakses (periksa GEMINI_API_KEY di .env)."
            return (
                pesan_err,
                "Fallback (Di Luar Cakupan)",
                skor_terbaik,
            )

        riwayat_percakapan.append((pesan_user, jawaban_llm))
        return jawaban_llm, "Gemini AI (RAG Terbatas)", skor_terbaik

    # JALUR 3: Skor terlalu rendah (< T_LOW) -> langsung jawab tidak tahu tanpa panggil LLM
    return PESAN_TIDAK_TAHU, "Fallback (Di Luar Cakupan)", skor_terbaik


# 8. Endpoint Flask
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True)
    if not data or "pesan" not in data or not str(data["pesan"]).strip():
        return jsonify({"error": "Pesan tidak boleh kosong"}), 400

    user_message = str(data["pesan"]).strip()
    respons, sumber, skor = get_response_hybrid(user_message)

    return jsonify({
        "respons": respons,
        "sumber": sumber,
        "skor": round(float(skor), 2),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
