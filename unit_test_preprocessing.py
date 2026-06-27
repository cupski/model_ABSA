import pandas as pd

from load_data import load_data
from preprocessing_functions import (
    clean_text,
    remove_emoji,
    lowercase,
    remove_url_mention,
    compress_repeated_chars,
    remove_special_chars,
    normalize_whitespace,
    normalize_slang,
    convert_labels,
    FINAL_ASPECTS,
)

# PENGUJIAN SAMPEL 
def test_pipeline_on_samples():
    """
    Uji setiap fungsi prapemrosesan pada data sampel kecil yang
    merepresentasikan karakteristik noise umum, sebelum memproses
    seluruh dataset.
    """
    print("=" * 60)
    print("PENGUJIAN FUNGSI PADA SAMPEL DATA")
    print("=" * 60)

    cases = [
        ("remove_emoji",            remove_emoji,            "aplikasi bagus banget \U0001f62d\U0001f62d"),
        ("lowercase",               lowercase,               "Aplikasi INI Sangat BAGUS"),
        ("remove_url_mention",      remove_url_mention,      "cek https://kompas.id dan @kompas untuk info"),
        ("compress_repeated_chars", compress_repeated_chars, "mahaaaaaal banget, lambaaaat sekali"),
        ("remove_special_chars",    remove_special_chars,    "ui/ux sangat ##$$ buruk, bikin pusing!!!"),
        ("normalize_whitespace",    normalize_whitespace,    "  terlalu   banyak   spasi   "),
        ("normalize_slang",         normalize_slang,         "gak bs dibuka, tp konten bgt bagus yg penting"),
    ]

    for name, fn, sample in cases:
        result = fn(sample)
        print(f"\n[{name}]")
        print(f"  Input : {sample}")
        print(f"  Output: {result}")

    # ── Uji clean_text (pipeline lengkap) ──────────────────────────
    noisy_samples = [
        "appsnya gak bisa dibuka sama sekali!!!! \U0001f62d\U0001f62d",
        "langganan mahal bgt tp konten kurang bagus",
        "https://kompas.id @kompas app bagus sekali",
        "UI/UX nya sangat ##$$ buruk bikin pusing!!!",
        "mahaaaaaal banget tp worth it sih",
    ]
    print("\n[clean_text — pipeline lengkap]")
    for s in noisy_samples:
        print(f"  Input : {s}")
        print(f"  Output: {clean_text(s)}")
        print()

    # ── Uji convert_labels pada DataFrame sampel ────────────────────
    sample_df = pd.DataFrame({
        'Komentar'              : ['bagus', 'kurang', 'biasa'],
        'Content Quality'       : [1, -1, None],
        'Subscription & Pricing': [0, None, -1],
        'UI/UX'                 : [None, 0, 1],
        'Functionality'         : [1, -1, None],
        'Technical & Access'    : [-1, None, 1],
    })
    labeled_df = convert_labels(sample_df.copy())
    print("[convert_labels — DataFrame sampel]")
    label_cols = ['Komentar'] + [f'lbl_{asp}' for asp in FINAL_ASPECTS]
    print(labeled_df[label_cols].to_string(index=False))
    print()

    print("=" * 60)
    print("Pengujian sampel selesai.")
    print("=" * 60)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_pipeline_on_samples()