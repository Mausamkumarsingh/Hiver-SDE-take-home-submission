from __future__ import annotations
from enum import Enum
from typing import Dict, List, Any
from pydantic import BaseModel

class IntentName(str, Enum):
    ORDER_TRACKING_AND_DELIVERY = "ORDER_TRACKING_AND_DELIVERY"
    RETURNS_AND_EXCHANGES = "RETURNS_AND_EXCHANGES"
    REFUND_AND_BILLING = "REFUND_AND_BILLING"
    DAMAGED_OR_DEFECTIVE_ITEM = "DAMAGED_OR_DEFECTIVE_ITEM"
    CANCELLATION_REQUEST = "CANCELLATION_REQUEST"
    ACCOUNT_ACCESS_AND_SECURITY = "ACCOUNT_ACCESS_AND_SECURITY"
    SUBSCRIPTION_AND_PRIME = "SUBSCRIPTION_AND_PRIME"
    DIGITAL_SERVICES_AND_DEVICES = "DIGITAL_SERVICES_AND_DEVICES"
    PRODUCT_AVAILABILITY_AND_PRICING = "PRODUCT_AVAILABILITY_AND_PRICING"
    CUSTOMER_SERVICE_COMPLAINT = "CUSTOMER_SERVICE_COMPLAINT"

class IntentDefinition(BaseModel):
    name: IntentName
    description: str
    keywords: List[str]
    exemplars: List[str]
    auto_handle_eligible: bool
    escalation_triggers: List[str]
    default_policy: str

TAXONOMY: Dict[IntentName, IntentDefinition] = {
    IntentName.ORDER_TRACKING_AND_DELIVERY: IntentDefinition(
        name=IntentName.ORDER_TRACKING_AND_DELIVERY,
        description="Inquiries regarding package shipping status, delivery dates, tracking numbers, and delivery delays.",
        keywords=["track", "tracking", "package", "delivery", "arrive", "shipped", "carrier", "late", "where is", "transit"],
        exemplars=[
            "Where is my package? It was supposed to arrive yesterday.",
            "My tracking number says delivered but there is no package on my porch.",
            "Can you tell me the estimated delivery date for my order?"
        ],
        auto_handle_eligible=True,
        escalation_triggers=["Package marked delivered but missing / stolen", "Delay exceeds 48 hours without tracking update"],
        default_policy="Provide order tracking portal link and carrier self-service guidance. Escalate if delivery failure or stolen package suspected."
    ),
    IntentName.RETURNS_AND_EXCHANGES: IntentDefinition(
        name=IntentName.RETURNS_AND_EXCHANGES,
        description="Questions about return policy, initiating a product return, return shipping labels, drop-off locations, and exchanges.",
        keywords=["return", "exchange", "drop off", "ups drop", "return label", "send back", "return window", "replacement"],
        exemplars=[
            "How do I return an item that doesn't fit?",
            "Can I get a return shipping label for my recent order?",
            "What is the return window for electronics?"
        ],
        auto_handle_eligible=True,
        escalation_triggers=["Item outside standard return window", "Drop-off barcode or label generation error"],
        default_policy="Auto-handle with official Your Orders return portal instructions. Escalate if manual exception or return fee dispute."
    ),
    IntentName.REFUND_AND_BILLING: IntentDefinition(
        name=IntentName.REFUND_AND_BILLING,
        description="Inquiries about refund processing timeline, duplicate charges, unexpected credit card debits, or payment method issues.",
        keywords=["refund", "charge", "charged", "overcharged", "bank", "credit card", "billed", "money back", "duplicate"],
        exemplars=[
            "I returned my package a week ago, when will I receive my refund?",
            "I was charged twice for order 104-982173. Please fix this.",
            "Why did you charge my credit card $14.99 without permission?"
        ],
        auto_handle_eligible=False,
        escalation_triggers=["Financial billing discrepancy", "Refund missing past 5-7 business days", "Unrecognized debit"],
        default_policy="Escalate to human finance/billing specialist. Bot cannot access private financial ledgers or issue banking adjustments."
    ),
    IntentName.DAMAGED_OR_DEFECTIVE_ITEM: IntentDefinition(
        name=IntentName.DAMAGED_OR_DEFECTIVE_ITEM,
        description="Reports of items arriving broken, smashed, defective, expired, or missing critical parts.",
        keywords=["damaged", "broken", "defective", "smashed", "ruined", "cracked", "missing parts", "doesn't work"],
        exemplars=[
            "The ceramic mug arrived completely shattered inside the box.",
            "The television screen is cracked and won't turn on.",
            "Box was crushed and two items were missing from my shipment."
        ],
        auto_handle_eligible=False,
        escalation_triggers=["Physical product damage", "Safety hazard or shattered glass", "High value defective electronics"],
        default_policy="Escalate for replacement authorization or photo evidence review by a human support specialist."
    ),
    IntentName.CANCELLATION_REQUEST: IntentDefinition(
        name=IntentName.CANCELLATION_REQUEST,
        description="Requests to cancel an order, accidental purchase, or cancel an item prior to carrier dispatch.",
        keywords=["cancel", "canceling", "cancel order", "cancel accidental", "stop order", "abort purchase"],
        exemplars=[
            "I ordered the wrong size by mistake, please cancel order immediately.",
            "Can I cancel my order before it ships?",
            "Need to cancel an order placed 10 minutes ago."
        ],
        auto_handle_eligible=True,
        escalation_triggers=["Item already entered shipping process and cannot be auto-cancelled in portal"],
        default_policy="Auto-handle with direct cancellation link if within cancellation window. Escalate if already shipped."
    ),
    IntentName.ACCOUNT_ACCESS_AND_SECURITY: IntentDefinition(
        name=IntentName.ACCOUNT_ACCESS_AND_SECURITY,
        description="Issues logging in, 2FA/OTP authentication errors, compromised or hacked accounts, and password resets.",
        keywords=["login", "password", "locked out", "otp", "2fa", "hacked", "unauthorized access", "compromised", "reset password"],
        exemplars=[
            "My account is locked and I cannot receive the OTP code on my phone.",
            "Someone hacked into my account and changed the delivery address!",
            "I forgot my password and the reset link is not arriving."
        ],
        auto_handle_eligible=False,
        escalation_triggers=["Account takeover / fraud suspicion", "2FA failure", "Locked account requiring identity verification"],
        default_policy="Escalate immediately to Account Security Specialist. Never attempt automated handling on compromised accounts."
    ),
    IntentName.SUBSCRIPTION_AND_PRIME: IntentDefinition(
        name=IntentName.SUBSCRIPTION_AND_PRIME,
        description="Inquiries regarding Prime membership, auto-renewal charges, student membership, Prime benefits, and membership cancellation.",
        keywords=["prime", "membership", "subscription", "annual fee", "prime video", "student prime", "renew", "cancel prime"],
        exemplars=[
            "I was charged $139 for Prime renewal, I want to cancel and get refunded.",
            "How do I sign up for the Prime student discount?",
            "What benefits are included in my Amazon Prime subscription?"
        ],
        auto_handle_eligible=True,
        escalation_triggers=["Refund request on accidental renewal with partial benefit usage", "Unauthorized subscription sign-up"],
        default_policy="Auto-handle with official Prime membership management link. Escalate if disputed renewal refund."
    ),
    IntentName.DIGITAL_SERVICES_AND_DEVICES: IntentDefinition(
        name=IntentName.DIGITAL_SERVICES_AND_DEVICES,
        description="Technical troubleshooting for FireStick, Kindle, Echo/Alexa, Prime Video streaming bugs, and app crashes.",
        keywords=["firestick", "kindle", "echo", "alexa", "app crashing", "prime video", "audio out of sync", "streaming error"],
        exemplars=[
            "Prime Video gives error code 5004 whenever I try to stream movies.",
            "My Fire TV Stick is stuck in a reboot loop and won't turn on.",
            "Kindle books are not syncing to my iPad app."
        ],
        auto_handle_eligible=True,
        escalation_triggers=["Hardware failure requiring warranty RMA", "Persistent firmware brick"],
        default_policy="Auto-handle with standard troubleshooting steps (restart, cache clear, network restart, app update)."
    ),
    IntentName.PRODUCT_AVAILABILITY_AND_PRICING: IntentDefinition(
        name=IntentName.PRODUCT_AVAILABILITY_AND_PRICING,
        description="Questions regarding item stock status, restock schedules, price matching, lightning deals, and coupon codes.",
        keywords=["in stock", "out of stock", "restock", "price match", "coupon", "promo code", "discount", "sale price"],
        exemplars=[
            "When will the Sony WH-1000XM5 headphones be back in stock?",
            "Do you price match if an item went on sale right after I bought it?",
            "My promo code for $10 off is showing as invalid at checkout."
        ],
        auto_handle_eligible=True,
        escalation_triggers=["Systemic promo code malfunction affecting paid orders", "Price guarantee dispute"],
        default_policy="Auto-handle with pricing/stock policies and promo terms. Escalate if manual checkout credit needed."
    ),
    IntentName.CUSTOMER_SERVICE_COMPLAINT: IntentDefinition(
        name=IntentName.CUSTOMER_SERVICE_COMPLAINT,
        description="Formal grievances, complaints about unhelpful representatives, severe delivery dissatisfaction, and supervisor escalation requests.",
        keywords=["horrible service", "rude agent", "worst customer service", "unacceptable", "supervisor", "manager", "disgusted", "lawyer"],
        exemplars=[
            "Your phone support representative hung up on me. I want a supervisor now.",
            "This is the third time your delivery driver threw my package in the rain. Horrible service!",
            "I have been waiting 3 weeks for an answer. This service is completely unacceptable."
        ],
        auto_handle_eligible=False,
        escalation_triggers=["Explicit demand for human supervisor", "Extreme frustration / abusive customer experience"],
        default_policy="Escalate immediately with an empathetic acknowledgment to a Senior Support Supervisor."
    )
}

ALL_INTENTS: List[str] = [i.value for i in IntentName]
