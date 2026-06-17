#!/usr/bin/env python3
import json, os

OUT = "/Users/lukai/Downloads/神山古典部_分心的练习_v2.json"

# Read the old card
with open('/Users/lukai/Downloads/神山古典部_分心的练习_千反田爱瑠_伊原摩耶花.json') as f:
    card = json.load(f)

# Fix: add spec for V2
card["spec"] = "chara_card_v2"

# Remove alternate_greetings (can cause import issues)
if "alternate_greetings" in card.get("extensions", {}):
    del card["extensions"]["alternate_greetings"]

# Keep only the first_mes, no alternate greetings
card["character_version"] = "1.0"

# Save
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(card, f, ensure_ascii=False, indent=2)

# Validate
with open(OUT) as f:
    data = json.load(f)

wb = data["extensions"]["world_book"]["entries"]
fm = data["first_mes"]

print(f"OK:{OUT}")
print(f"Size: {os.path.getsize(OUT):,} bytes")
print(f"Entries: {len(wb)}")
print(f"first_mes: {len(fm)} chars")
print(f"spec: {data.get('spec', 'MISSING')}")
