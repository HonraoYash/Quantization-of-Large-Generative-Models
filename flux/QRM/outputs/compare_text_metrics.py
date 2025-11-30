
import argparse, os, glob, json, re
def load(path): return open(path, 'r', encoding='utf-8').read().strip()
def toks(s):     return re.findall(r"\w+|\S", s.lower())
def distinct_n(tokens, n):
    if len(tokens) < n: return 0.0
    grams = list(zip(*[tokens[i:] for i in range(n)]))
    return len(set(grams)) / max(1, len(grams))
def try_bleu(ref, hyp):
    try:
        import sacrebleu
        return float(sacrebleu.corpus_bleu([hyp], [[ref]]).score)
    except Exception: return None
def try_rougeL(ref, hyp):
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(ref, hyp)['rougeL']
        # return F1 as a % for readability
        return 100.0 * sc.fmeasure
    except Exception: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='directory with out_*.txt files')
    ap.add_argument('--ref', default='out_fp16.txt', help='reference filename inside --dir')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "out_*.txt")))
    if not files:
        raise SystemExit("No out_*.txt files in " + args.dir)

    ref_path = os.path.join(args.dir, args.ref)
    ref_txt = load(ref_path) if os.path.exists(ref_path) else None

    rows, summary = [], {}
    for f in files:
        name = os.path.basename(f)
        txt = load(f)
        tk  = toks(txt)
        d1, d2 = distinct_n(tk,1), distinct_n(tk,2)
        row = {
            "file": name,
            "chars": len(txt),
            "words": len([t for t in tk if re.match(r"\w+", t)]),
            "distinct1": round(d1,4),
            "distinct2": round(d2,4),
        }
        if ref_txt is not None and name != args.ref:
            bleu = try_bleu(ref_txt, txt)
            rL   = try_rougeL(ref_txt, txt)
            if bleu is not None: row["BLEU"] = round(bleu,2)
            if rL   is not None: row["ROUGE-L_F1"] = round(rL,2)
        rows.append(row)
        summary[name] = row

    # pretty print
    cols = ["file","chars","words","distinct1","distinct2","BLEU","ROUGE-L_F1"]
    print(" | ".join(c.ljust(14) for c in cols))
    print("-"*90)
    for r in rows:
        print(" | ".join(str(r.get(c,"")).ljust(14) for c in cols))

    # save json
    with open(os.path.join(args.dir,"compare_metrics.json"),"w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved:", os.path.join(args.dir,"compare_metrics.json"))
if __name__ == "__main__":
    main()

