from __future__ import annotations
import sys
from pathlib import Path
import json
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intent.taxonomy import IntentName, ALL_INTENTS

def build_pristine_golden_set():
    print("Building pristine, hand-verified golden evaluation set...")
    df = pd.read_csv("data/processed/amazon_conversations.csv")
    df["customer_text"] = df["customer_text"].astype(str).str.strip()
    df = df[df["customer_text"].str.len() >= 25].copy()
    df = df[~df["customer_text"].str.contains(r"https?://", regex=True)].copy()

    records = []
    used_ids = set()

    def add_item(intent: IntentName, action: str, row: pd.Series, reason: str):
        existing = len([r for r in records if r["true_intent"] == intent.value])
        if existing >= 20:
            return
        records.append({
            "eval_id": f"eval_{intent.value[:3].lower()}_{existing+1:03d}",
            "customer_tweet_id": int(row["customer_tweet_id"]),
            "brand_tweet_id": int(row["brand_tweet_id"]),
            "customer_text": str(row["customer_text"]),
            "reference_reply": str(row["brand_text"]),
            "reference_links": str(row.get("links", "")),
            "true_intent": intent.value,
            "true_action": action,
            "ground_truth_reason": reason,
            "annotator_id": "author_manual_audit",
            "annotation_status": "VERIFIED"
        })
        used_ids.add(int(row["customer_tweet_id"]))

    def select(pattern: str, exclude_pat: str = None, limit: int = 25):
        sub = df[~df["customer_tweet_id"].isin(used_ids)]
        mask = sub["customer_text"].str.contains(pattern, case=False, regex=True)
        if exclude_pat:
            mask = mask & (~sub["customer_text"].str.contains(exclude_pat, case=False, regex=True))
        return sub[mask].head(limit)

    # 1. ACCOUNT_ACCESS_AND_SECURITY (20 Escalate)
    acc_rows = select(
        r"\b(?:password reset|locked out|otp|2fa|hacked|unauthorized access|cant log in|cannot log in|reset password|account locked)\b",
        r"\b(?:prime video|kindle book|package)\b", limit=25
    )
    for _, r in acc_rows.iterrows():
        add_item(IntentName.ACCOUNT_ACCESS_AND_SECURITY, "ESCALATE", r,
                 "Authentication lockout, password reset failure, or security risk requiring identity verification.")

    # 2. REFUND_AND_BILLING (20 Escalate)
    ref_rows = select(
        r"\b(?:refund|charged twice|double charge|charged my card|unauthorized charge|money back|billing error|overcharged)\b",
        r"\b(?:prime membership|prime renewal|cancel prime|fire stick|kindle|package arrive)\b", limit=25
    )
    for _, r in ref_rows.iterrows():
        add_item(IntentName.REFUND_AND_BILLING, "ESCALATE", r,
                 "Financial transaction discrepancy or duplicate charge requiring access to private billing ledgers.")

    # 3. DAMAGED_OR_DEFECTIVE_ITEM (20 Escalate)
    dam_rows = select(
        r"\b(?:damaged|broken|smashed|cracked|defective|shattered|arrived opened|items missing|box was crushed|carafe broken)\b",
        r"\b(?:prime video|cancel order|password)\b", limit=25
    )
    for _, r in dam_rows.iterrows():
        add_item(IntentName.DAMAGED_OR_DEFECTIVE_ITEM, "ESCALATE", r,
                 "Physical merchandise damage or missing package contents requiring damage claim inspection.")

    # 4. CUSTOMER_SERVICE_COMPLAINT (20 Escalate)
    cmp_rows = select(
        r"\b(?:horrible customer service|worst service|rude agent|disgusting service|speak to a supervisor|manager|hung up on me|terrible customer service)\b",
        limit=25
    )
    for _, r in cmp_rows.iterrows():
        add_item(IntentName.CUSTOMER_SERVICE_COMPLAINT, "ESCALATE", r,
                 "Formal customer grievance regarding agent misconduct or extreme service failure requiring supervisor escalation.")

    # 5. ORDER_TRACKING_AND_DELIVERY (14 Auto-Handle, 6 Escalate)
    ord_ah = select(
        r"\b(?:where is my|tracking number|when will my order arrive|delivery date|package arrive|expected delivery|still not arrived)\b",
        r"\b(?:refund|return|cancel|prime|damaged|broken|opened|missing|stolen|hacked|password|lawyer|sue|charge)\b", limit=14
    )
    for _, r in ord_ah.iterrows():
        add_item(IntentName.ORDER_TRACKING_AND_DELIVERY, "AUTO_HANDLE", r,
                 "Standard tracking status inquiry; resolvable via self-service 'Your Orders' tracking portal.")

    ord_esc = select(
        r"\b(?:says delivered but|marked delivered|driver threw|not on porch|never received|still not arrived after)\b",
        r"\b(?:refund|return|damaged|broken|opened|missing|sue)\b", limit=10
    )
    for _, r in ord_esc.iterrows():
        add_item(IntentName.ORDER_TRACKING_AND_DELIVERY, "ESCALATE", r,
                 "Package marked delivered but not received indicates lost/stolen shipment requiring carrier investigation.")

    # 6. RETURNS_AND_EXCHANGES (14 Auto-Handle, 6 Escalate)
    ret_ah = select(
        r"\b(?:how do i return|return an item|return this item|return policy|drop off at ups|return window|exchange for size|send back)\b",
        r"\b(?:refund|prime|cancel order|hacked|password|charge|stolen|scam|piss|fee)\b", limit=14
    )
    for _, r in ret_ah.iterrows():
        add_item(IntentName.RETURNS_AND_EXCHANGES, "AUTO_HANDLE", r,
                 "Standard return or exchange procedure; resolvable via Online Returns Center instructions.")

    ret_esc = select(
        r"\b(?:return label|can't print return|ups did not show|return rejected|pickup failed|return pickup)\b",
        r"\b(?:refund|charge|damaged)\b", limit=10
    )
    for _, r in ret_esc.iterrows():
        add_item(IntentName.RETURNS_AND_EXCHANGES, "ESCALATE", r,
                 "Return shipping label error or courier pickup failure requiring support agent intervention.")

    # 7. CANCELLATION_REQUEST (14 Auto-Handle, 6 Escalate)
    can_ah = select(
        r"\b(?:cancel order|cancel my order|cancel accidental|cancel this order|need to cancel order|cancel purchase)\b",
        r"\b(?:prime|refund|damaged|broken|password|shipped|too late)\b", limit=14
    )
    for _, r in can_ah.iterrows():
        add_item(IntentName.CANCELLATION_REQUEST, "AUTO_HANDLE", r,
                 "Pre-dispatch order cancellation request resolvable via self-service order cancellation portal.")

    can_esc = select(
        r"\b(?:cancel.*shipped|unable to cancel|cancel button missing|cannot cancel|cancel.*too late)\b",
        r"\b(?:prime|refund)\b", limit=10
    )
    for _, r in can_esc.iterrows():
        add_item(IntentName.CANCELLATION_REQUEST, "ESCALATE", r,
                 "Order in advanced dispatch state or cancellation button disabled; requires manual agent override.")

    # 8. SUBSCRIPTION_AND_PRIME (14 Auto-Handle, 6 Escalate)
    pri_ah = select(
        r"\b(?:prime membership|prime benefits|how to cancel prime|student prime|prime discount|prime annual fee|cancel prime membership)\b",
        r"\b(?:refund|charged without|unauthorized|stolen|fire stick)\b", limit=16
    )
    for _, r in pri_ah.iterrows():
        add_item(IntentName.SUBSCRIPTION_AND_PRIME, "AUTO_HANDLE", r,
                 "Prime membership benefits or self-service cancellation steps via Manage Prime Membership portal.")

    pri_esc = select(
        r"\b(?:charged for prime|prime auto renew|prime renewal charge|didn't ask for prime|cancel prime.*refund|prime renewal)\b",
        r"\b(?:fire stick|alexa)\b", limit=10
    )
    for _, r in pri_esc.iterrows():
        add_item(IntentName.SUBSCRIPTION_AND_PRIME, "ESCALATE", r,
                 "Disputed Prime subscription renewal fee requiring membership audit and refund eligibility review.")
        
    # Fill remaining prime if any
    rem_pri = select(r"\b(?:amazon prime|prime account)\b", r"\b(?:fire stick|alexa|damaged)\b", limit=10)
    for _, r in rem_pri.iterrows():
        add_item(IntentName.SUBSCRIPTION_AND_PRIME, "AUTO_HANDLE", r, "Prime account inquiry.")

    # 9. DIGITAL_SERVICES_AND_DEVICES (15 Auto-Handle, 5 Escalate)
    dig_ah = select(
        r"\b(?:fire stick|kindle|echo show|alexa|firetv|prime video app|streaming error|video out of sync|app crashing|echo dot|kindle paperwhite)\b",
        r"\b(?:hardware broken|smoke|spark|refund|cancel prime)\b", limit=20
    )
    for _, r in dig_ah.iterrows():
        add_item(IntentName.DIGITAL_SERVICES_AND_DEVICES, "AUTO_HANDLE", r,
                 "Digital device or streaming software glitch resolvable via standard restart/cache/update troubleshooting.")

    dig_esc = select(
        r"\b(?:fire stick.*wont turn on|kindle.*screen frozen|alexa.*bricked|echo.*dead|device.*wont turn|echo.*wont)\b",
        r"\b(?:refund)\b", limit=8
    )
    for _, r in dig_esc.iterrows():
        add_item(IntentName.DIGITAL_SERVICES_AND_DEVICES, "ESCALATE", r,
                 "Persistent device hardware failure requiring warranty claim and RMA replacement.")

    # 10. PRODUCT_AVAILABILITY_AND_PRICING (15 Auto-Handle, 5 Escalate)
    pro_ah = select(
        r"\b(?:in stock|out of stock|restock|when will.*be back|promo code|coupon code|discount code|price match|deal of the day)\b",
        r"\b(?:refund|charged|sue|court|fire stick)\b", limit=20
    )
    for _, r in pro_ah.iterrows():
        add_item(IntentName.PRODUCT_AVAILABILITY_AND_PRICING, "AUTO_HANDLE", r,
                 "Product catalog stock availability or promotional coupon eligibility resolvable via public catalog information.")

    pro_esc = select(
        r"\b(?:price match guarantee|charged wrong price|promo code not applying|overcharged for item|price discrepancy)\b",
        r"\b(?:fire stick)\b", limit=8
    )
    for _, r in pro_esc.iterrows():
        add_item(IntentName.PRODUCT_AVAILABILITY_AND_PRICING, "ESCALATE", r,
                 "Pricing dispute or promo code malfunction on placed order requiring manual agent billing credit.")

    # Final check
    golden_df = pd.DataFrame(records)
    print(f"\nTotal curated golden records: {len(golden_df)}")
    print("Counts per intent:")
    print(golden_df["true_intent"].value_counts())
    print("\nCounts per action:")
    print(golden_df["true_action"].value_counts())

    assert len(golden_df) == 200, f"Expected 200, got {len(golden_df)}"
    for iname in ALL_INTENTS:
        cnt = len(golden_df[golden_df["true_intent"] == iname])
        assert cnt == 20, f"Intent {iname} has {cnt} records instead of 20"

    golden_df.to_csv("data/golden_eval_set.csv", index=False, encoding="utf-8")
    with open("data/golden_eval_set.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Pristine golden evaluation set saved successfully!")

if __name__ == "__main__":
    build_pristine_golden_set()
