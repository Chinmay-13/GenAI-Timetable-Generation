import pandas as pd, json

# --- room_assignment.csv ---
df = pd.read_csv("outputs/room_assignment.csv")
print("=== room_assignment.csv ===")
print("Shape:", df.shape)
print("Cols:", list(df.columns))
print("Days:", df["Day"].unique().tolist())
print("Periods:", sorted(df["Period"].unique().tolist()))
print("Rooms sample:", df["Room"].unique()[:10].tolist())
print("Section count:", df["Section"].nunique())
print("ROOM_UNASSIGNED count:", (df["Room"] == "ROOM_UNASSIGNED").sum())

# Tuesday P2 specifically
t2 = df[(df["Day"] == "Tuesday") & (df["Period"] == "P2")]
print("\nTuesday P2 rows:", len(t2))
print(t2[["Section","Course","Room"]].to_string())

# --- rag_docs.json ---
print("\n=== rag_docs.json ===")
docs = json.load(open("outputs/rag_docs.json", encoding="utf-8"))
sections = set(d.get("section") for d in docs if "section" in d)
print("Total docs:", len(docs))
print("Sections indexed:", sorted(s for s in sections if s))
print("Faculty docs:", sum(1 for d in docs if "faculty" in d))
print("Section docs:", sum(1 for d in docs if "section" in d))
print("Room docs:", sum(1 for d in docs if "room" in d.get("source","").lower()))
