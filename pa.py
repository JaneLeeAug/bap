import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eva_speaker", default="eva_speaker.csv")
    parser.add_argument("--tts_sentence", default="tts_sentence.txt")
    parser.add_argument("--output_csv", default="eva_pairing_table.csv")
    parser.add_argument("--num_sentences_per_speaker", type=int, default=40)
    parser.add_argument("--num_speakers", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 讀 speaker
    speaker_df = pd.read_csv(args.eva_speaker, dtype=str)

    required_cols = ["speaker_id", "transcript", "wavFile"]
    for col in required_cols:
        if col not in speaker_df.columns:
            raise ValueError(f"eva_speaker.csv missing column: {col}")

    # 只保留 wavFile 裡面有 commonvoice 的 speaker
    speaker_df = speaker_df[
        speaker_df["wavFile"].str.contains("commonvoice", case=False, na=False)
    ].copy()

    # 同一個 speaker_id 只留一列
    speaker_df = speaker_df.drop_duplicates(subset=["speaker_id"])

    # 抽樣 60 個 speaker
    if len(speaker_df) < args.num_speakers:
        raise ValueError(
            f"commonvoice speaker 不足 {args.num_speakers} 個，目前只有 {len(speaker_df)} 個"
        )

    speaker_df = speaker_df.sample(
        n=args.num_speakers,
        random_state=args.seed
    ).reset_index(drop=True)

    # 讀句子
    with open(args.tts_sentence, "r", encoding="utf-8") as f:
        sentences = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # 去重，保留順序
    sentences = list(dict.fromkeys(sentences))

    total_needed = len(speaker_df) * args.num_sentences_per_speaker

    if total_needed > len(sentences):
        raise ValueError(
            f"句子不足，需要 {total_needed} 句，但 tts_sentence.txt 只有 {len(sentences)} 句"
        )

    rows = []
    sentence_idx = 0
    output_idx = 1

    for _, speaker in speaker_df.iterrows():
        selected = sentences[
            sentence_idx:
            sentence_idx + args.num_sentences_per_speaker
        ]

        sentence_idx += args.num_sentences_per_speaker

        for sentence in selected:
            rows.append({
                "speaker_id": speaker["speaker_id"],
                "speaker_prompt_text_transcription": speaker["transcript"],
                "speaker_prompt_audio_filename": speaker["wavFile"],
                "content_to_synthesize": sentence,
                "output_audio_filename": f"eva{output_idx:04d}",
            })

            output_idx += 1

    out_df = pd.DataFrame(rows)

    out_df.to_csv(
        args.output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"saved: {args.output_csv}")
    print(f"speaker count: {len(speaker_df)}")
    print(f"unique sentences: {len(sentences)}")
    print(f"sentences per speaker: {args.num_sentences_per_speaker}")
    print(f"generated rows: {len(out_df)}")


if __name__ == "__main__":
    main()
