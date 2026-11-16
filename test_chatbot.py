"""
Skrip Pengujian Chatbot Hybrid dengan RAG Terbatas (Modul 5 & 6)
Menguji:
1. Perbandingan akurasi Intent Matching: Rule-Based vs Hybrid TF-IDF (termasuk kasus Typo & Sinonim)
2. Pengambilan Top-K Dokumen Konteks (dengan Sinonim & Variasi Kalimat)
3. Pemilihan Jalur Hybrid 3 Jalur, Memori Percakapan Multi-Turn, & Guardrail "Tidak Tahu"
"""

import app as chatbot


def uji_perbandingan_intent():
    print("=" * 75)
    print(f"1. PENGUJIAN INTENT MATCHING: RULE-BASED vs TF-IDF (T_HIGH = {chatbot.T_HIGH:.2f})")
    print("=" * 75)

    test_cases = [
        ("halo selamat pagi", "greeting"),
        ("siapa rektor unp saat ini", "rektor_unp","Rektor of unp"),
        ("siapa ketua kampus unp", "rektor_unp"),              # Uji sinonim 'ketua kampus' -> rektor
        ("dimana cek jadwla perkuliahan", "jadwal_kuliah"),    # Uji typo 'jadwla'
        ("letak kampus utama unp dimana", "lokasi_kampus"),    # Uji sinonim 'letak' -> lokasi
        ("cara hubungi admin akademik kampus", "kontak_admin","Admin contact"),
        ("terima kasih atas bantuannya", "terima_kasih"),
        ("sampai jumpa lagi", "goodbye"),
        ("bagaimana cuaca di kota padang hari ini", None),
    ]

    benar_rule = 0
    benar_tfidf = 0

    for teks, expected in test_cases:
        hasil_rule = chatbot.match_intent_rule_based(teks)
        intent_raw, _, skor = chatbot.match_intent_tfidf(teks)
        hasil_tfidf = intent_raw if skor >= chatbot.T_HIGH else None

        ok_rule = hasil_rule == expected
        ok_tfidf = hasil_tfidf == expected
        if ok_rule:
            benar_rule += 1
        if ok_tfidf:
            benar_tfidf += 1

        status_r = "BENAR" if ok_rule else "SALAH"
        status_t = "BENAR" if ok_tfidf else "SALAH"
        print(
            f"Input : '{teks}'\n"
            f"  Target     : {expected}\n"
            f"  Rule-Based : [{status_r}] -> {hasil_rule}\n"
            f"  TF-IDF     : [{status_t}] -> {hasil_tfidf} (skor={skor:.2f})"
        )

    total = len(test_cases)
    print("-" * 75)
    print(f"Akurasi Rule-Based : {benar_rule / total * 100:.1f}% ({benar_rule}/{total})")
    print(f"Akurasi TF-IDF     : {benar_tfidf / total * 100:.1f}% ({benar_tfidf}/{total})\n")


def uji_top_k_retrieval():
    print("=" * 75)
    print(f"2. PENGUJIAN RETRIEVAL TOP-K DOKUMEN + SINONIM (k = {chatbot.TOP_K})")
    print("=" * 75)

    queries = [
        "bagaimana cara mengajukan berhenti studi sementara di unp?",  # sinonim -> cuti akademik
        "syarat ambil sks dan bayar uang semesteran di siakad",        # sinonim -> krs & ukt
        "daftar fakultas yang ada di universitas negeri padang",
        "tidak bisa login akun portal siakad karena lupa password",    # sinonim -> kendala login
    ]

    for q in queries:
        skor_terbaik, docs = chatbot.cari_top_k(q, k=chatbot.TOP_K)
        judul_list = [f"{d['id']} ({d['skor']:.2f})" for d in docs]
        print(f"Query : '{q}'")
        print(f"  Skor Terbaik : {skor_terbaik:.2f}")
        print(f"  Top-K Docs   : {', '.join(judul_list)}")
    print()


def uji_router_dan_memori():
    print("=" * 75)
    print("3. PENGUJIAN ROUTER 3 JALUR, MEMORI MULTI-TURN, & GUARDRAIL 'TIDAK TAHU'")
    print("=" * 75)

    chatbot.riwayat_percakapan.clear()

    skenario = [
        ("Cocok KB (Jalur 1)", "siapa rektor unp", "Knowledge Base Lokal (TF-IDF)"),
        ("Typo + Sinonim KB (Jalur 1)", "dimana letak kmpus utama unp", "Knowledge Base Lokal (TF-IDF)"),
        (
            "Kompleks di Dokumen (Jalur 2)",
            "bagaimana prosedur mengajukan cuti akademik bagi mahasiswa?",
            "Gemini AI (RAG Terbatas) / Fallback bila API Key belum valid",
        ),
        (
            "Pertanyaan Lanjutan Multi-Turn (Memori)",
            "persetujuan siapa yang diperlukan untuk itu?",
            "Gemini AI (RAG Terbatas) — Konteks mengikuti topik sebelumnya",
        ),
        (
            "Konteks Kurang (Wajib Tidak Tahu)",
            "berapa biaya ukt jurusan kedokteran tahun 2029?",
            "Fallback (Di Luar Cakupan)",
        ),
        (
            "Di Luar Topik (Jalur 3)",
            "resep nasi goreng spesial enak",
            "Fallback (Di Luar Cakupan)",
        ),
        (
            "Jebakan Pengetahuan Umum (Anti-Halusinasi)",
            "siapa presiden pertama republik indonesia?",
            "Fallback (Di Luar Cakupan)",
        ),
    ]

    for label_skenario, pesan, ekspektasi in skenario:
        respons, sumber, skor = chatbot.get_response_hybrid(pesan)
        print(f"[{label_skenario}]")
        print(f"  Pesan      : {pesan}")
        print(f"  Ekspektasi : {ekspektasi}")
        print(f"  Sumber     : {sumber} (skor={skor:.2f})")
        print(f"  Respons    : {respons}\n")


if __name__ == "__main__":
    uji_perbandingan_intent()
    uji_top_k_retrieval()
    uji_router_dan_memori()
