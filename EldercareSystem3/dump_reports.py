"""Read all per-clip reports and dump to a single UTF8 file for inspection."""
import os

output = []
for fn in sorted(os.listdir("output3/llm_reports/per_clip")):
    fp = os.path.join("output3/llm_reports/per_clip", fn)
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    output.append("=" * 60)
    output.append(f"FILE: {fn}")
    output.append("=" * 60)
    output.append(content[:2000])
    output.append("")

with open("per_clip_dump.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Done. Output written to per_clip_dump.txt")
