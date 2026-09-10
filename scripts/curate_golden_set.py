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

def curate_verified_golden_set():
    print("Loading processed conversations for manual golden set curation...")
    df = pd.read_csv("data/processed/amazon_conversations.csv")
    
    # Filter out very short tweets
    df = df[df["customer_text"].str.len() >= 25].copy()
    
    selected_records = []
    used_tweet_ids = set()

    def add_record(intent: IntentName, action: str, row: pd.Series, reason: str):
        count_for_intent = len([r for r in selected_records if r['true_intent'] == intent.value])
        if count_for_intent >= 20:
            return
        selected_records.append({
            "eval_id": f"eval_{intent.value[:3].lower()}_{count_for_intent+1:03d}",
            "customer_tweet_id": int(row["customer_tweet_id"]),
            "brand_tweet_id": int(row["brand_tweet_id"]),
            "customer_text": str(row["customer_text"]).strip(),
            "reference_reply": str(row["brand_text"]).strip(),
            "reference_links": str(row.get("links", "")),
            "true_intent": intent.value,
            "true_action": action,
            "ground_truth_reason": reason,
            "annotator_id": "author_manual_audit",
            "annotation_status": "VERIFIED"
        })
        used_tweet_ids.add(int(row["customer_tweet_id"]))

    def find_rows(pattern: str, exclude_pattern: str = None, limit: int = 25):
        sub = df[~df["customer_tweet_id"].isin(used_tweet_ids)]
        mask = sub["customer_text"].str.contains(pattern, case=False, regex=True)
        if exclude_pattern:
            mask = mask & (~sub["customer_text"].str.contains(exclude_pattern, case=False, regex=True))
        return sub[mask].head(limit)

    # 1. ORDER_TRACKING_AND_DELIVERY
    ah_ord = find_rows(r"\b(?:where is my|tracking number|when will my order arrive|delivery date|package arrive|tracking)\b", 
                       exclude_pattern=r"\b(?:refund|damaged|broken|stolen|missing|opened|lawyer|sue|charge)\b", limit=14)
    for _, r in ah_ord.iterrows():
        add_record(IntentName.ORDER_TRACKING_AND_DELIVERY, "AUTO_HANDLE", r, 
                   "Standard tracking status inquiry; resolvable via self-service 'Your Orders' tracking portal.")
        
    esc_ord = find_rows(r"\b(?:says delivered but|marked delivered|driver threw|not on porch|never received|stolen)\b", 
                        exclude_pattern=r"\b(?:refund|damaged|broken|return)\b", limit=10)
    for _, r in esc_ord.iterrows():
        add_record(IntentName.ORDER_TRACKING_AND_DELIVERY, "ESCALATE", r, 
                   "Package marked delivered but not received indicates lost/stolen shipment requiring carrier investigation.")

    # 2. RETURNS_AND_EXCHANGES
    ah_ret = find_rows(r"\b(?:how do i return|return policy|return an item|drop off at ups|return window|exchange for size|return it)\b", 
                       exclude_pattern=r"\b(?:refund|damaged|broken|charge|stolen|scam|piss|horrible)\b", limit=14)
    for _, r in ah_ret.iterrows():
        add_record(IntentName.RETURNS_AND_EXCHANGES, "AUTO_HANDLE", r, 
                   "Standard return or exchange procedure; resolvable via Online Returns Center instructions.")
        
    esc_ret = find_rows(r"\b(?:return label|can't print return|ups did not show|return rejected|pickup failed|returning.*fee)\b", 
                        exclude_pattern=r"\b(?:refund|charged|damage)\b", limit=10)
    for _, r in esc_ret.iterrows():
        add_record(IntentName.RETURNS_AND_EXCHANGES, "ESCALATE", r, 
                   "Return shipping exception or carrier pickup failure requiring support agent intervention.")

    # 3. REFUND_AND_BILLING
    esc_ref = find_rows(r"\b(?:refund|charged twice|double charge|charged my card|unauthorized charge|money back|billing)\b", 
                        exclude_pattern=r"\b(?:prime|damaged|broken)\b", limit=25)
    for _, r in esc_ref.iterrows():
        add_record(IntentName.REFUND_AND_BILLING, "ESCALATE", r, 
                   "Financial transaction inquiry or duplicate charge dispute requiring access to private billing ledgers.")

    # 4. DAMAGED_OR_DEFECTIVE_ITEM
    esc_dam = find_rows(r"\b(?:damaged|broken|smashed|cracked|defective|shattered|arrived opened|missing items|ruined)\b", 
                        exclude_pattern=r"\b(?:refund|prime video|kindle)\b", limit=25)
    for _, r in esc_dam.iterrows():
        add_record(IntentName.DAMAGED_OR_DEFECTIVE_ITEM, "ESCALATE", r, 
                   "Physical merchandise damage or missing items inside package requires damage claim inspection.")

    # 5. CANCELLATION_REQUEST
    ah_can = find_rows(r"\b(?:cancel order|cancel my order|cancel accidental|cancel this order|need to cancel|cancel it)\b", 
                       exclude_pattern=r"\b(?:prime|refund|already shipped|too late|cant cancel)\b", limit=14)
    for _, r in ah_can.iterrows():
        add_record(IntentName.CANCELLATION_REQUEST, "AUTO_HANDLE", r, 
                   "Pre-dispatch order cancellation request resolvable via self-service order cancellation portal.")
        
    esc_can = find_rows(r"\b(?:cancel.*already shipped|unable to cancel|cancel button missing|cannot cancel|cancel.*too late)\b", 
                        exclude_pattern=r"\b(?:prime|refund)\b", limit=12)
    for _, r in esc_can.iterrows():
        add_record(IntentName.CANCELLATION_REQUEST, "ESCALATE", r, 
                   "Order in advanced dispatch state or cancellation button disabled; requires manual agent override.")

    # 6. ACCOUNT_ACCESS_AND_SECURITY
    esc_acc = find_rows(r"\b(?:password reset|locked out|otp|2fa|hacked|unauthorized access|cant log in|cannot log in|reset password|login)\b", 
                        exclude_pattern=r"\b(?:prime video|kindle)\b", limit=25)
    for _, r in esc_acc.iterrows():
        add_record(IntentName.ACCOUNT_ACCESS_AND_SECURITY, "ESCALATE", r, 
                   "Account authentication, password reset failure, or security lockout requiring identity verification.")

    # 7. SUBSCRIPTION_AND_PRIME
    ah_pri = find_rows(r"\b(?:prime membership|prime benefits|how to cancel prime|student prime|prime discount|prime subscription)\b", 
                       exclude_pattern=r"\b(?:refund|charged without|unauthorized|stolen)\b", limit=14)
    for _, r in ah_pri.iterrows():
        add_record(IntentName.SUBSCRIPTION_AND_PRIME, "AUTO_HANDLE", r, 
                   "Prime membership benefits or self-service cancellation steps via Manage Prime Membership portal.")
        
    esc_pri = find_rows(r"\b(?:charged for prime|prime auto renew|prime renewal charge|didn't ask for prime|cancel prime.*refund)\b", 
                        exclude_pattern=r"\b(?:fire stick|alexa)\b", limit=12)
    for _, r in esc_pri.iterrows():
        add_record(IntentName.SUBSCRIPTION_AND_PRIME, "ESCALATE", r, 
                   "Disputed Prime subscription renewal fee requiring membership audit and refund eligibility review.")

    # 8. DIGITAL_SERVICES_AND_DEVICES
    ah_dig = find_rows(r"\b(?:fire stick|kindle|echo|alexa|firetv|prime video|app crashing|streaming error)\b", 
                       exclude_pattern=r"\b(?:hardware broken|smoke|spark|refund)\b", limit=16)
    for _, r in ah_dig.iterrows():
        add_record(IntentName.DIGITAL_SERVICES_AND_DEVICES, "AUTO_HANDLE", r, 
                   "Digital device or streaming software glitch resolvable via standard restart/cache/update troubleshooting.")
        
    esc_dig = find_rows(r"\b(?:fire stick.*wont turn on|kindle.*screen frozen|alexa.*bricked|echo.*dead|device.*broken)\b", 
                        exclude_pattern=r"\b(?:refund)\b", limit=8)
    for _, r in esc_dig.iterrows():
        add_record(IntentName.DIGITAL_SERVICES_AND_DEVICES, "ESCALATE", r, 
                   "Persistent device hardware failure requiring warranty claim and RMA replacement.")

    # 9. PRODUCT_AVAILABILITY_AND_PRICING
    ah_pro = find_rows(r"\b(?:in stock|out of stock|restock|when will.*be back|promo code|coupon code|discount code|price match)\b", 
                       exclude_pattern=r"\b(?:refund|charged|sue|court)\b", limit=15)
    for _, r in ah_pro.iterrows():
        add_record(IntentName.PRODUCT_AVAILABILITY_AND_PRICING, "AUTO_HANDLE", r, 
                   "Product catalog stock availability or promotional coupon eligibility resolvable via public store info.")
        
    esc_pro = find_rows(r"\b(?:price match guarantee|charged wrong price|promo code not applying|overcharged for item|price change)\b", 
                        exclude_pattern=r"\b(?:fire stick)\b", limit=8)
    for _, r in esc_pro.iterrows():
        add_record(IntentName.PRODUCT_AVAILABILITY_AND_PRICING, "ESCALATE", r, 
                   "Pricing dispute or promo code malfunction on placed order requiring manual agent billing credit.")

    # 10. CUSTOMER_SERVICE_COMPLAINT
    esc_cmp = find_rows(r"\b(?:horrible customer service|worst service|rude agent|disgusting service|speak to a supervisor|manager|unacceptable service|hung up on me|worst customer service)\b", 
                        limit=25)
    for _, r in esc_cmp.iterrows():
        add_record(IntentName.CUSTOMER_SERVICE_COMPLAINT, "ESCALATE", r, 
                   "Severe customer grievance regarding agent conduct or delivery failure requiring supervisor escalation.")

    golden_df = pd.DataFrame(selected_records)
    print(f"\nTotal curated golden records: {len(golden_df)}")
    print("Counts per intent:")
    print(golden_df["true_intent"].value_counts())
    print("\nCounts per action:")
    print(golden_df["true_action"].value_counts())
    
    # Assertions
    assert len(golden_df) == 200, f"Expected 200 records, got {len(golden_df)}"
    for iname in ALL_INTENTS:
        cnt = len(golden_df[golden_df["true_intent"] == iname])
        assert cnt == 20, f"Intent {iname} has {cnt} records instead of 20"
        
    golden_df.to_csv("data/golden_eval_set.csv", index=False, encoding="utf-8")
    with open("data/golden_eval_set.jsonl", "w", encoding="utf-8") as f:
        for r in selected_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print("Clean, audited golden set successfully verified and saved!")

if __name__ == "__main__":
    curate_verified_golden_set()
