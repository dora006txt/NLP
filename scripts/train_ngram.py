import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nlp.common.artifacts import ArtifactPaths, ensure_dir
from nlp.ngram.kenlm_model import build_kenlm_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/wikitext-103/wiki.train.tokens")
    parser.add_argument("--limit-lines", type=int, default=None)
    parser.add_argument("--max-n", type=int, default=3)
    parser.add_argument("--memory", default="50%")
    parser.add_argument("--candidate-vocab-size", type=int, default=30000)
    parser.add_argument("--prune", nargs="*", type=int, default=None)
    parser.add_argument("--kenlm-root", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    artifacts = ArtifactPaths()
    out_path = args.out or os.path.join(artifacts.ngram_dir(), "kenlm_word.arpa.json")
    ensure_dir(os.path.dirname(out_path))

    build_kenlm_model(
        train_path=args.train_path,
        output_manifest_path=out_path,
        order=args.max_n,
        limit_lines=args.limit_lines,
        memory=args.memory,
        candidate_vocab_size=args.candidate_vocab_size,
        prune=args.prune,
        kenlm_root=args.kenlm_root,
    )
    print(out_path)


if __name__ == "__main__":
    main()
