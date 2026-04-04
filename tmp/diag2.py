import pandas as pd, json, sys
sys.path.insert(0, ".")

out = []

try:
    df = pd.read_csv("outputs/room_assignment.csv")
    out.append(f"Shape: {df.shape}")
    out.append(f"Cols: {list(df.columns)}")
    out.append(f"Days: {df['Day'].unique().tolist()}")
    out.append(f"Periods: {sorted(df['Period'].unique().tolist())}")
    out.append(f"Rooms sample: {df['Room'].unique()[:10].tolist()}")
    out.append(f"Section count: {df['Section'].nunique()}")
    out.append(f"ROOM_UNASSIGNED: {(df['Room']=='ROOM_UNASSIGNED').sum()}")
    t2 = df[(df["Day"]=="Tuesday") & (df["Period"]=="P2")]
    out.append(f"Tuesday P2 rows: {len(t2)}")
    for _, r in t2.iterrows():
        out.append(f"  {r['Section']} {r['Course']} => {r['Room']}")
except Exception as e:
    out.append(f"room_assignment error: {e}")

try:
    docs = json.load(open("outputs/rag_docs.json", encoding="utf-8"))
    secs = sorted(set(d.get("section") for d in docs if d.get("section")))
    out.append(f"RAG total docs: {len(docs)}")
    out.append(f"RAG sections: {secs}")
    out.append(f"Faculty docs: {sum(1 for d in docs if 'faculty' in d)}")
    out.append(f"Section docs: {sum(1 for d in docs if 'section' in d)}")
except Exception as e:
    out.append(f"rag_docs error: {e}")

open("tmp/diag_out.txt", "w", encoding="utf-8").write("\n".join(out))
print("DONE")
