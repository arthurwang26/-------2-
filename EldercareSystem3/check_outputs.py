"""Inspect all output3 debug/events/llm_reports to trace error sources."""
import json, os, glob

print("=" * 80)
print("PART 1: SKELETON ACTION MODEL RAW OUTPUT (debug/*/actions.json)")
print("=" * 80)

for actions_file in sorted(glob.glob("output3/debug/*/actions.json")):
    clip_name = os.path.basename(os.path.dirname(actions_file))
    with open(actions_file, "r", encoding="utf-8") as f:
        actions = json.load(f)
    print(f"\n--- {clip_name} ---")
    for tid_str, act_list in actions.items():
        for act in act_list:
            a = act["action"]
            c = act["confidence"]
            sf = act["start_frame"]
            ef = act["end_frame"]
            print(f"  TID={tid_str}: {a} (conf={c:.2f}) frames {sf}->{ef}")

print("\n" + "=" * 80)
print("PART 2: EVENTS SENT TO LLM (debug/events.json)")
print("=" * 80)

with open("output3/debug/events.json", "r", encoding="utf-8") as f:
    events = json.load(f)

for clip_name, ev_list in events.items():
    print(f"\n--- {clip_name} ---")
    for ev in ev_list:
        person = ev.get("person", "?")
        etype = ev.get("type", "?")
        action = ev.get("action", "?")
        conf = ev.get("confidence", 0)
        sf = ev.get("start_frame", "?")
        ef = ev.get("end_frame", "?")
        blip = ev.get("blip_verified", None)
        print(f"  {person}: {etype}={action} (conf={conf:.2f}) frames {sf}->{ef}" + (f" blip={blip}" if blip is not None else ""))

print("\n" + "=" * 80)
print("PART 3: CLIP HOI EVENTS (debug/*/clip_hoi.json)")
print("=" * 80)

for hoi_file in sorted(glob.glob("output3/debug/*/clip_hoi.json")):
    clip_name = os.path.basename(os.path.dirname(hoi_file))
    with open(hoi_file, "r", encoding="utf-8") as f:
        hoi_events = json.load(f)
    # Filter HOI-type events only
    hoi_only = [h for h in hoi_events if h.get("type") in ("HOI", "HOI-CLIP")]
    if hoi_only:
        print(f"\n--- {clip_name} ---")
        for h in hoi_only:
            person = h.get("person", "?")
            action = h.get("action", "?")
            conf = h.get("confidence", 0)
            blip = h.get("blip_verified", None)
            print(f"  {person}: HOI={action} (conf={conf:.2f})" + (f" blip={blip}" if blip is not None else ""))

print("\n" + "=" * 80)
print("PART 4: LLM REPORT CONTENT (system_docs/final_report.md)")
print("=" * 80)

report_path = "output3/system_docs/final_report.md"
if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read()[:3000])

print("\n" + "=" * 80)
print("PART 5: GROUND TRUTH COMPARISON")
print("=" * 80)

gt_path = "output3/debug/ground_truth.json"
if os.path.exists(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    print(json.dumps(gt, ensure_ascii=False, indent=2)[:2000])
else:
    # Check for ground truth in other locations
    for p in glob.glob("**/ground_truth*", recursive=True):
        print(f"Found GT file: {p}")
    for p in glob.glob("**/groundtruth*", recursive=True):
        print(f"Found GT file: {p}")
    for p in glob.glob("**/gt_*", recursive=True):
        print(f"Found GT file: {p}")
    print("No ground truth file found in output3/debug/")
