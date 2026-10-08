"""
generate_data.py
-----------------
Generates a realistic synthetic dataset of customer support tickets for the
AI Customer Support Ticket Triage project.

Each ticket has:
    - ticket_text : the raw text a customer would type
    - category     : billing | technical | account | product
    - urgency      : low | medium | high | critical

The text is built from templates + slot-filling so that the language is varied
(the same way real ticket text varies) while keeping the labels controlled,
which lets us measure the pipeline properly.
"""

import random
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# 1. Building blocks for each category
# ---------------------------------------------------------------------------

PRODUCTS = ["the mobile app", "your website", "the desktop client", "the API",
            "the Chrome extension", "the billing portal", "the dashboard",
            "the Android app", "the iOS app", "the admin panel", "the reporting module",
            "the notifications service", "the onboarding wizard", "the settings page"]

# Extra slot vocabulary used to multiply template variety far beyond the raw
# template count (see fill_template).
ISSUES = ["a blank white screen", "a spinning loader that never finishes",
          "a timeout error", "a memory leak that freezes my browser tab",
          "duplicate entries", "a broken image thumbnail", "garbled text",
          "an unresponsive submit button", "a permissions error",
          "data that resets after every refresh"]
FEATURES = ["dark mode", "a bulk-import tool", "single sign-on (SSO)",
            "offline mode", "keyboard shortcuts", "a mobile widget",
            "custom dashboard layouts", "scheduled email reports",
            "multi-language support", "a Slack integration",
            "role-based permissions", "an audit log", "a REST API webhook",
            "drag-and-drop file uploads", "a dark theme with adjustable font size"]
ACCOUNT_ACTIONS = ["reset my password", "change my registered email",
                    "unlock my account", "merge two of my accounts",
                    "transfer workspace ownership", "delete my account permanently",
                    "correct my company name", "add a new team member",
                    "verify my identity", "update my billing contact",
                    "recover a deactivated account", "change my username"]

BILLING_TEMPLATES = [
    "I was charged {amount} twice on my {card} card for the same {plan} subscription.",
    "My invoice for {month} shows {amount} but I was quoted a different price when I signed up.",
    "I cancelled my {plan} plan last week but you still billed me {amount} today.",
    "Can you explain why there is an extra {amount} charge on my latest invoice?",
    "I need a refund of {amount} because I was double billed for {plan}.",
    "The auto-renewal charged my card {amount} even though I turned it off.",
    "Please update my payment method, my {card} card on file has expired.",
    "I upgraded to {plan} but I'm still being billed at the old rate.",
    "Where can I download my GST invoice for the {month} payment of {amount}?",
    "I would like to downgrade my subscription and get a prorated refund.",
    "Your system charged me in USD instead of INR, the amount {amount} looks wrong.",
    "I never authorized a payment of {amount}, please investigate this charge.",
    "My {card} card was declined but you still show my {plan} plan as active, please clarify billing.",
    "I was quoted {amount}/month for {plan} but the {month} invoice shows a higher figure.",
    "Can I get a corrected invoice for {month}? The tax calculation on {amount} looks off.",
    "I switched from {plan} to a cheaper plan but the {amount} charge didn't change.",
    "Please cancel my {card} card auto-debit for the {plan} subscription starting {month}.",
    "There are two separate {amount} charges for {plan} on the same day, is this a billing glitch?",
    "I'd like a breakdown of what the {amount} charge on my {month} invoice actually covers.",
    "My team's {plan} plan renewal of {amount} went through despite our cancellation request.",
]

TECHNICAL_TEMPLATES = [
    "{product} keeps crashing every time I try to open the reports section.",
    "I'm getting a 500 error whenever I upload a file larger than 10MB to {product}.",
    "{product} is stuck on the loading screen and won't respond after the latest update.",
    "Login fails with 'invalid token' error even though my password is correct on {product}.",
    "The API is returning a 429 rate-limit error much earlier than the documented limit.",
    "None of my dashboard charts are loading in {product}, the page just shows a blank space.",
    "I keep getting logged out of {product} every few minutes on Safari.",
    "The export to CSV feature in {product} produces a corrupted file.",
    "Push notifications from {product} stopped working after I updated my phone's OS.",
    "There is a sync issue, changes made on {product} do not reflect on the mobile app.",
    "The search bar in {product} throws a JavaScript error in the console.",
    "Two-factor authentication codes are not being delivered for {product} logins.",
    "{product} shows {issue} whenever I try to load my saved reports.",
    "After the latest release, {product} has {issue} on every page load.",
    "Can someone look into {issue} on {product}? It started this morning.",
    "I'm seeing {issue} in {product} only when I use Google Chrome, not Firefox.",
    "{product} threw {issue} right after I connected my third-party integration.",
    "Every time I try to save changes in {product}, I get {issue}.",
    "Support ticket: {product} has {issue}, restarting the app doesn't help.",
    "Is anyone else seeing {issue} on {product} today, or is it just my account?",
]

ACCOUNT_TEMPLATES = [
    "I can't reset my password, the reset link for my account never arrives.",
    "Please help me change the email address linked to my account.",
    "My account got locked after too many login attempts, can you unlock it?",
    "I want to merge two accounts that were created with different emails.",
    "How do I transfer ownership of my organization's workspace to a new admin?",
    "I need to permanently delete my account and all associated data.",
    "My profile shows the wrong company name, how can I correct it?",
    "I'm not able to add new team members, it says my account has reached the seat limit.",
    "Can you verify my identity, I think someone else is trying to access my account?",
    "I never received the verification email after signing up.",
    "My subscription is active but the account still shows as 'free tier'.",
    "I need to update the billing contact on my enterprise account.",
    "Could you help me {action}? I've tried everything on my end.",
    "I need to {action} but the option seems to be missing from settings.",
    "Support, please assist me to {action} as soon as you're able to.",
    "I'm trying to {action} for my organization's account but keep hitting an error.",
    "Is it possible to {action} without losing my existing data and history?",
    "I emailed last week asking to {action} and haven't heard back yet.",
    "Can an admin on your side help me {action}? Self-service isn't working.",
    "What's the process to {action} for a team of 15 people?",
]

PRODUCT_TEMPLATES = [
    "It would be great if {product} supported dark mode on tablets.",
    "Can you add an option to export reports directly to Google Sheets from {product}?",
    "I'd like to request a bulk-import feature for {product}.",
    "Is there a roadmap for adding offline mode to {product}?",
    "The onboarding tutorial for {product} is confusing, some steps are missing.",
    "Could you add keyboard shortcuts to speed up navigation in {product}?",
    "I think {product} should let us customize the dashboard layout.",
    "Please consider adding multi-language support to {product}.",
    "A dark theme and adjustable font size would really help with accessibility on {product}.",
    "Is there any plan to integrate {product} with Slack?",
    "The mobile version of {product} feels very different from the desktop one, any plans to align them?",
    "Suggestion: let us schedule recurring reports to be emailed automatically from {product}.",
    "Would it be possible to add {feature} to {product}? It would save our team a lot of time.",
    "Feature request: {feature} for {product}, several of my colleagues have asked for this too.",
    "Any chance {feature} is on the roadmap for {product} this year?",
    "We'd really benefit from {feature} in {product} for our enterprise workflow.",
    "Just a suggestion — {feature} would make {product} much easier to use daily.",
    "Our team would love to see {feature} added to {product} in a future release.",
    "Is {feature} something you're considering for {product}? Curious about the roadmap.",
    "Small idea: {feature} in {product} could really improve onboarding for new users.",
]

RESOLUTIONS = {
    "billing": [
        "Refund issued to original payment method within 5-7 business days.",
        "Invoice corrected and resent to customer; duplicate charge reversed.",
        "Payment method updated by agent; subscription reactivated at correct rate.",
        "Escalated to billing team for manual review of tax calculation.",
    ],
    "technical": [
        "Issue reproduced and fixed in the next patch release; customer notified.",
        "Cleared cache/cookies resolved the issue; steps sent to customer.",
        "Escalated to engineering as a known bug; workaround provided in the meantime.",
        "Root cause traced to a third-party integration outage; resolved once vendor restored service.",
    ],
    "account": [
        "Identity verified via support call; account access restored.",
        "Email updated after verification steps; confirmation sent to new address.",
        "Accounts merged by internal admin tooling; customer confirmed data intact.",
        "Escalated to account security team due to suspicious login attempts.",
    ],
    "product": [
        "Logged as a feature request in the product backlog for roadmap review.",
        "Shared with the product team; added to the Q3 planning discussion.",
        "Not currently planned; alternative workaround suggested to customer.",
        "Upvoted internally after multiple similar requests from other customers.",
    ],
}

CATEGORY_TEMPLATES = {
    "billing": BILLING_TEMPLATES,
    "technical": TECHNICAL_TEMPLATES,
    "account": ACCOUNT_TEMPLATES,
    "product": PRODUCT_TEMPLATES,
}

# Urgency phrasing is layered on top, independent of category, so the model
# genuinely has to learn urgency language rather than memorizing category.
URGENCY_PREFIXES = {
    "critical": [
        "URGENT: ", "This is critical, ", "Emergency - ", "Production is down, ",
        "This needs immediate attention, ",
    ],
    "high": [
        "This is quite urgent, ", "Please prioritize this, ", "I need this resolved today, ",
        "This is affecting my whole team, ",
    ],
    "medium": [
        "", "When you get a chance, ", "Not blocking but important, ",
    ],
    "low": [
        "No rush, ", "Just a minor thing, ", "Whenever convenient, ", "Small question - ",
    ],
}

URGENCY_SUFFIXES = {
    "critical": [
        " We are losing customers because of this right now.",
        " All of our staff are unable to work because of this.",
        " This has completely stopped our operations.",
    ],
    "high": [
        " This is blocking an important deadline for us.",
        " We have a client demo tomorrow and this must be fixed.",
        " Several users have already complained about this.",
    ],
    "medium": [
        " Please let me know an approximate timeline.",
        " It's been happening for a couple of days now.",
        "",
    ],
    "low": [
        " Take your time, just wanted to flag it.",
        " Not a big deal, just curious.",
        "",
    ],
}

AMOUNTS = ["₹499", "₹999", "₹1,999", "₹4,999", "$9.99", "$29.99", "$49.00", "₹15,000"]
CARDS = ["Visa", "Mastercard", "RuPay", "American Express"]
PLANS = ["Basic", "Pro", "Premium", "Enterprise", "Team"]
MONTHS = ["January", "March", "June", "August", "September", "November"]

# Category -> plausible urgency distribution (not uniform, mirrors real life:
# billing/account issues skew medium/high, technical skews across the board,
# product requests skew low).
URGENCY_WEIGHTS = {
    "billing":   {"low": 0.15, "medium": 0.45, "high": 0.30, "critical": 0.10},
    "technical": {"low": 0.10, "medium": 0.30, "high": 0.35, "critical": 0.25},
    "account":   {"low": 0.10, "medium": 0.35, "high": 0.40, "critical": 0.15},
    "product":   {"low": 0.55, "medium": 0.35, "high": 0.08, "critical": 0.02},
}


def fill_template(template: str) -> str:
    return template.format(
        product=random.choice(PRODUCTS),
        amount=random.choice(AMOUNTS),
        card=random.choice(CARDS),
        plan=random.choice(PLANS),
        month=random.choice(MONTHS),
        issue=random.choice(ISSUES),
        feature=random.choice(FEATURES),
        action=random.choice(ACCOUNT_ACTIONS),
    )


def sample_urgency(category: str) -> str:
    weights = URGENCY_WEIGHTS[category]
    return random.choices(list(weights.keys()), weights=list(weights.values()))[0]


# Common typos / shorthand real users type, applied at low frequency so the
# vocabulary isn't perfectly clean (mirrors real support inboxes).
TYPO_SWAPS = {
    "the": "teh", "account": "acount", "please": "pls", "because": "becuase",
    "immediately": "immediatly", "subscription": "subscrption", "receive": "recieve",
    "separate": "seperate", "cancelled": "canceled",
}

# A short "cross-talk" clause that can be stitched onto a ticket from a
# *different* category, to create the kind of ambiguous, multi-topic tickets
# that make real ticket triage hard (and stop the model from just memorizing
# a closed template list).
CROSS_TALK = {
    "billing": " Also, the app crashed right after the payment screen.",
    "technical": " By the way I also noticed an odd charge on my last invoice.",
    "account": " This started right after I updated my billing details.",
    "product": " On a related note, I was also charged more than expected this month.",
}


def apply_typos(text: str) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        bare = w.strip(".,!?'\"").lower()
        if bare in TYPO_SWAPS and random.random() < 0.5:
            words[i] = w.lower().replace(bare, TYPO_SWAPS[bare])
    return " ".join(words)


def build_ticket(category: str) -> dict:
    template = random.choice(CATEGORY_TEMPLATES[category])
    body = fill_template(template)
    urgency = sample_urgency(category)
    prefix = random.choice(URGENCY_PREFIXES[urgency])
    suffix = random.choice(URGENCY_SUFFIXES[urgency])

    body = body[0].upper() + body[1:]
    text = f"{prefix}{body}{suffix}".strip()

    # ~18% of tickets get a cross-topic clause stitched on -> genuine ambiguity
    # between two categories, the way real tickets often blend issues.
    if random.random() < 0.18:
        text = text + CROSS_TALK[category]

    # ~10% of tickets get light typos, mimicking real user typing
    if random.random() < 0.10:
        text = apply_typos(text)

    # light noise to mimic real typing (extra spaces / missing capitalisation)
    if random.random() < 0.08:
        text = text.lower()
    if random.random() < 0.05:
        text = text.replace("  ", " ")

    return {
        "ticket_text": text,
        "category": category,
        "urgency": urgency,
        "resolution_notes": random.choice(RESOLUTIONS[category]),
    }


def generate_dataset(n_per_category: int = 220, category_noise_rate: float = 0.04,
                      urgency_noise_rate: float = 0.05) -> pd.DataFrame:
    rows = []
    for category in CATEGORY_TEMPLATES:
        for _ in range(n_per_category):
            rows.append(build_ticket(category))
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.insert(0, "ticket_id", [f"TCK-{i+1:05d}" for i in range(len(df))])

    # --- Simulate real-world annotation noise: a small % of tickets get a
    # mislabeled category, the way a rushed human tagger occasionally would. ---
    all_categories = list(CATEGORY_TEMPLATES.keys())
    n_noisy = int(len(df) * category_noise_rate)
    noisy_idx = df.sample(n=n_noisy, random_state=7).index
    for idx in noisy_idx:
        current = df.at[idx, "category"]
        choices = [c for c in all_categories if c != current]
        df.at[idx, "category"] = random.choice(choices)

    # --- Same idea for urgency: agents often disagree on how urgent something
    # is (e.g. one agent calls it "high", another "critical"), so a small %
    # of urgency labels are nudged to a neighbouring level rather than being
    # perfectly consistent with the phrasing used. ---
    urgency_order = ["low", "medium", "high", "critical"]
    n_urg_noisy = int(len(df) * urgency_noise_rate)
    urg_noisy_idx = df.sample(n=n_urg_noisy, random_state=11).index
    for idx in urg_noisy_idx:
        current = df.at[idx, "urgency"]
        pos = urgency_order.index(current)
        # nudge to an adjacent urgency level (agents rarely confuse "low" with "critical")
        neighbors = [p for p in (pos - 1, pos + 1) if 0 <= p < len(urgency_order)]
        df.at[idx, "urgency"] = urgency_order[random.choice(neighbors)]

    return df


if __name__ == "__main__":
    df = generate_dataset(n_per_category=5000)
    out_path = "data/support_tickets.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} tickets -> {out_path}")
    print(f"Unique ticket_text values: {df['ticket_text'].nunique()} / {len(df)}")
    print(df["category"].value_counts())
    print(df["urgency"].value_counts())
    print(df.head(3).to_string())
