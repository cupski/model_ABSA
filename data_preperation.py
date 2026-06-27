import os
import json
import argparse
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
    stratified_split,
    check_split_size,
    compare_distribution,
    compute_class_weights,
    FINAL_ASPECTS,
    NUM_CLASSES,
    LABEL_NAMES,
)


# ── PIPELINE UTAMA ────────────────────────────────────────────────────────────

def run_pipeline(data_path: str, output_dir: str = '.'):
    """
    Menjalankan seluruh tahapan prapemrosesan secara berurutan:
      1. Baca dataset berlabel
      2. Pembersihan teks
      3. Konversi label
      4. Stratified split 70/15/15
      5. Hitung class weights
      6. Simpan hasil ke output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DATA PREPARATION — ABSA Kompas.id")
    print(f"{'='*60}")
    print(f"Input : {data_path}")
    print(f"Output: {output_dir}/\n")

    # Step 1 — Baca dataset
    df = load_data(data_path)
    print(f"Total data: {len(df)} baris")

    none_counts = {
        asp: int(df[asp].isna().sum())
        for asp in FINAL_ASPECTS if asp in df.columns
    }
    print("NaN per aspek (akan → kelas None):")
    for asp, cnt in none_counts.items():
        print(f"  {asp}: {cnt}")
    print(f"Total baris: {len(df)} (tidak ada yang dihapus)")

    # Step 2 — Pembersihan teks
    print("\n[1/4] Preprocessing teks...")
    df['komentar_clean'] = df['Komentar'].apply(clean_text)
    empty = (df['komentar_clean'].str.strip() == '').sum()
    if empty > 0:
        df.loc[df['komentar_clean'].str.strip() == '', 'komentar_clean'] = (
            df.loc[df['komentar_clean'].str.strip() == '', 'Komentar']
            .str.lower().str.strip()
        )
    print(f"  Selesai | Teks kosong setelah clean: {empty} (diisi fallback)")

    # Step 3 — Konversi label
    print("\n[2/4] Konversi label...")
    df = convert_labels(df)
    for asp in FINAL_ASPECTS:
        col      = f'lbl_{asp}'
        n_cls    = NUM_CLASSES[asp]
        none_cnt = int((df[col] == (n_cls - 1)).sum())
        print(f"  {asp}: {len(df)} baris | {n_cls} kelas | None={none_cnt}")
    print("  Selesai")

    # Step 4 — Stratified split
    print("\n[3/4] Stratified split 70/15/15...")
    df_train, df_val, df_test = stratified_split(df)
    check_split_size(df_train, df_val, df_test)
    aspect_cols = [f'lbl_{asp}' for asp in FINAL_ASPECTS]
    for col in aspect_cols:
        compare_distribution(df_train, df_val, df_test, col)
    print("  Selesai")

    # Step 5 — Class weights
    print("\n[4/4] Menghitung class weights...")
    cw = compute_class_weights(df_train)
    for asp, w in cw.items():
        lnames = LABEL_NAMES[asp]
        detail = ' | '.join(f"{ln}:{wi:.4f}" for ln, wi in zip(lnames, w))
        print(f"  {asp}: {detail}")
    print("  Selesai")

    # Simpan semua output
    df_train.to_csv(f'{output_dir}/df_train.csv')
    df_val.to_csv(f'{output_dir}/df_val.csv')
    df_test.to_csv(f'{output_dir}/df_test.csv')

    with open(f'{output_dir}/class_weights.json', 'w') as f:
        json.dump(cw, f, indent=2, ensure_ascii=False)

    stats = {
        'total': len(df),
        'train': len(df_train),
        'val'  : len(df_val),
        'test' : len(df_test),
        'aspects': {
            asp: {
                'num_classes'  : NUM_CLASSES[asp],
                'label_names'  : LABEL_NAMES[asp],
                'class_weights': cw[asp],
                'train_dist'   : [int((df_train[f'lbl_{asp}'] == c).sum())
                                  for c in range(NUM_CLASSES[asp])],
                'val_dist'     : [int((df_val[f'lbl_{asp}'] == c).sum())
                                  for c in range(NUM_CLASSES[asp])],
                'test_dist'    : [int((df_test[f'lbl_{asp}'] == c).sum())
                                  for c in range(NUM_CLASSES[asp])],
            } for asp in FINAL_ASPECTS
        }
    }
    with open(f'{output_dir}/data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("DATA PREPARATION SELESAI")
    print(f"{'='*60}")
    print(f"File tersimpan di: {output_dir}/")
    print(f"  df_train.csv       -> {len(df_train)} baris training")
    print(f"  df_val.csv         -> {len(df_val)} baris validasi")
    print(f"  df_test.csv        -> {len(df_test)} baris test")
    print(f"  class_weights.json -> bobot kelas per aspek")
    print(f"  data_stats.json    -> statistik lengkap dataset")

    return df_train, df_val, df_test, cw


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

    # ── Uji fungsi teks atom ────────────────────────────────────────
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
    parser = argparse.ArgumentParser(
        description='Data preparation untuk training ABSA Kompas.id')
    parser.add_argument('--data_path',  type=str, default='ABSA_dataset_final_CLEAN.csv',
                        help='Path ke file CSV dataset berlabel')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Direktori output untuk file hasil preparation')
    parser.add_argument('--test_only',  action='store_true',
                        help='Hanya jalankan pengujian sampel, tanpa memproses dataset penuh')
    args = parser.parse_args()

    test_pipeline_on_samples()

    if not args.test_only:
        run_pipeline(args.data_path, args.output_dir)
