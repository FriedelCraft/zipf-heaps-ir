"""
Download the FULL Bengali (ben_Beng) split of AI4Bharat/IndicCorpV2 -- no
sampling, all ~29.6M rows. Streams row-by-row and writes directly to disk,
so memory use stays flat regardless of corpus size.

Install first:
    pip install datasets

Usage:
    python download_corpus.py --out_raw bn_indiccorp_full_raw.txt \
        --out_meta bn_corpus_metadata_full.json
"""

import argparse
import json
import time
from datetime import datetime, timezone

from datasets import load_dataset

VALID_SPLITS = {
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
    "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva", "mal_Mlym", "mar_Deva",
    "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "snd_Deva",
    "tam_Taml", "tel_Telu", "urd_Arab", "khasi", "santhali",
}


def count_complete_lines(path):
    """Count fully-written lines in an existing output file, and truncate
    off any trailing partial line (in case the process died mid-write).
    Returns the number of complete rows already saved -- this is exactly
    how many rows we need to skip when resuming the stream."""
    import os
    if not os.path.exists(path):
        return 0

    with open(path, 'rb') as f:
        f.seek(0, 2)  # end of file
        size = f.tell()
        if size == 0:
            return 0
        f.seek(-1, 2)
        ends_with_newline = (f.read(1) == b'\n')

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    complete_lines = lines if ends_with_newline else lines[:-1]

    if not ends_with_newline and len(lines) > 0:
        # rewrite the file without the dangling partial last line
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(complete_lines)
        print(f"  (dropped 1 incomplete trailing line from a previous interrupted run)")

    return len(complete_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='ben_Beng',
                     help=f'IndicCorpV2 split = language. Valid: {sorted(VALID_SPLITS)}')
    ap.add_argument('--text_field', default='text')
    ap.add_argument('--out_raw', default='bn_indiccorp_full_raw.txt')
    ap.add_argument('--out_meta', default='bn_corpus_metadata_full.json')
    ap.add_argument('--progress_every', type=int, default=500_000)
    ap.add_argument('--max_rows', type=int, default=None,
                     help='stop after this many total rows are saved (e.g. 10000000). '
                          'Leave unset to stream the entire split.')
    args = ap.parse_args()

    if args.split not in VALID_SPLITS:
        raise ValueError(f"'{args.split}' not recognized. Valid: {sorted(VALID_SPLITS)}")

    already_have = count_complete_lines(args.out_raw)
    if already_have > 0:
        print(f"Found existing partial download: {already_have:,} rows already saved in {args.out_raw}")
        print(f"Resuming -- will skip the first {already_have:,} rows of the stream.")
    else:
        print("No existing partial file found, starting fresh.")

    if args.max_rows is not None and already_have >= args.max_rows:
        print(f"Already have {already_have:,} rows, which meets/exceeds --max_rows={args.max_rows:,}. "
              f"Nothing more to download -- writing metadata only.")
        total_rows = already_have
        empty_rows = 0
        elapsed = 0.0
        stopped_reason = "cap_already_met"
    else:

        print(f"Opening AI4Bharat/IndicCorpV2 [config='indiccorp_v2', split='{args.split}'] "
              f"in streaming mode -- this will take a while.")
        ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split=args.split,
                           streaming=True, encoding="utf-8", encoding_errors="replace")

        if already_have > 0:
            ds = ds.skip(already_have)

        total_rows = already_have
        empty_rows = 0
        start = time.time()
        file_mode = 'a' if already_have > 0 else 'w'
        stopped_reason = "completed_split"

        try:
            with open(args.out_raw, file_mode, encoding='utf-8') as f:
                for row in ds:
                    text = row.get(args.text_field, "")
                    if not text:
                        empty_rows += 1
                        continue
                    f.write(text.replace('\n', ' ').strip() + '\n')
                    total_rows += 1

                    if total_rows % args.progress_every == 0:
                        elapsed = time.time() - start
                        new_rows = total_rows - already_have
                        rate = new_rows / elapsed if elapsed > 0 else 0
                        remaining_target = args.max_rows if args.max_rows is not None else 29_600_000
                        remaining_est = (remaining_target - total_rows) / rate if rate > 0 else float('nan')
                        print(f"  ...{total_rows:,} total rows saved ({elapsed/60:.1f} min this session, "
                              f"{rate:.0f} rows/s) -- est. {remaining_est/60:.1f} min remaining")

                    if args.max_rows is not None and total_rows >= args.max_rows:
                        stopped_reason = "max_rows_cap_reached"
                        print(f"\nReached --max_rows cap of {args.max_rows:,}. Stopping cleanly.")
                        break
        except Exception as e:
            print(f"\n--- Stream interrupted: {type(e).__name__}: {e} ---")
            print(f"Progress saved: {total_rows:,} rows are safely on disk in {args.out_raw}")
            print("Just re-run this same script (with the same --max_rows) to resume -- nothing is lost.")
            return

        elapsed = time.time() - start

    print(f"\nDone. {total_rows:,} rows written ({empty_rows:,} empty rows skipped) "
          f"in {elapsed/60:.1f} minutes.")
    print(f"Saved: {args.out_raw}")

    ESTIMATED_FULL_SPLIT_ROWS = 29_600_000  # as reported on the dataset page for ben_Beng
    pct_of_full = round(100 * total_rows / ESTIMATED_FULL_SPLIT_ROWS, 1)

    if args.max_rows is not None:
        sampling_desc = (
            f"Capped download: streamed sequentially and stopped after "
            f"{args.max_rows:,} rows (--max_rows). Reason for stopping: {stopped_reason}. "
            f"This is {pct_of_full}% of the full ben_Beng split "
            f"(~{ESTIMATED_FULL_SPLIT_ROWS:,} rows)."
        )
    else:
        sampling_desc = (
            f"Full split streamed with no cap. Reason for stopping: {stopped_reason}. "
            f"{pct_of_full}% of the estimated full split size."
        )

    metadata = {
        "dataset": "ai4bharat/IndicCorpV2",
        "config_name": "indiccorp_v2",
        "split_used": args.split,
        "language": "Bengali",
        "script": "Bengali (Beng)",
        "source_file_in_repo": "data/bn.txt",
        "download_date_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_method": sampling_desc,
        "max_rows_cap": args.max_rows,
        "total_rows_written": total_rows,
        "estimated_full_split_size": ESTIMATED_FULL_SPLIT_ROWS,
        "percent_of_full_split": pct_of_full,
        "empty_rows_skipped": empty_rows,
        "download_time_minutes": round(elapsed / 60, 1),
        "text_field_used": args.text_field,
    }
    with open(args.out_meta, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {args.out_meta}")
    print("\n--- Record these in your methodology doc ---")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()