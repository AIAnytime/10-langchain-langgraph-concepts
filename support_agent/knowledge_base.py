"""
A tiny in-memory knowledge base for the support agent to retrieve from.

In production this would be a vector database (Pinecone, pgvector, Chroma...)
holding millions of documents. For a self-contained tutorial we keep a handful
of realistic help-center articles in memory so the demo runs offline with no
extra services — the *retrieval pattern* (concept 5) is identical either way.
"""

# Each article: an id, a title, the searchable body, and metadata for filtering.
KNOWLEDGE_BASE = [
    {
        "id": "KB-001",
        "title": "Payment Retry Policy",
        "category": "billing",
        "text": (
            "If a payment fails we automatically retry it after 24 and 72 hours. "
            "A failed payment is usually caused by an expired card, insufficient "
            "funds, or a bank hold. Customers can update their card under "
            "Settings > Billing and click 'Retry now' to charge immediately."
        ),
    },
    {
        "id": "KB-002",
        "title": "Refund Eligibility",
        "category": "billing",
        "text": (
            "Refunds are available within 30 days of purchase for annual plans and "
            "within 7 days for monthly plans. Enterprise contracts follow the terms "
            "in the signed order form. Refunds return to the original payment method "
            "within 5-10 business days."
        ),
    },
    {
        "id": "KB-003",
        "title": "Resetting Your Password",
        "category": "technical",
        "text": (
            "To reset a password, click 'Forgot password' on the login page and "
            "follow the emailed link, which expires in 60 minutes. If the email does "
            "not arrive, check spam and confirm the account email address is correct."
        ),
    },
    {
        "id": "KB-004",
        "title": "API Rate Limits and 429 Errors",
        "category": "technical",
        "text": (
            "The API allows 60 requests per minute on Free, 600 on Pro, and custom "
            "limits on Enterprise. Exceeding the limit returns HTTP 429. Use "
            "exponential backoff and read the Retry-After header to recover cleanly."
        ),
    },
    {
        "id": "KB-005",
        "title": "Plan Tiers and Pricing",
        "category": "sales",
        "text": (
            "We offer Free, Pro ($29/mo), and Enterprise (custom) plans. Pro adds "
            "higher rate limits, priority support, and team seats. Enterprise adds "
            "SSO, an SLA, a dedicated success manager, and volume discounts."
        ),
    },
    {
        "id": "KB-006",
        "title": "Upgrading or Downgrading Your Plan",
        "category": "sales",
        "text": (
            "You can change plans anytime under Settings > Billing. Upgrades take "
            "effect immediately with prorated billing. Downgrades take effect at the "
            "end of the current billing cycle so you keep paid features until then."
        ),
    },
    {
        "id": "KB-007",
        "title": "Data Export and Account Deletion",
        "category": "general",
        "text": (
            "You can export all your data as JSON or CSV from Settings > Privacy. "
            "Account deletion is permanent and removes data within 30 days, per our "
            "retention policy. Contact support to request an expedited deletion."
        ),
    },
]
