# What is Compliance Copilot?

**It fills in questionnaires using your own documents, and tells you which questions
your documents can't answer.**

---

## The problem

Your company is closing a deal. The client's procurement team sends a **40-question
security questionnaire** and won't sign until it's answered.

Someone now spends **two days** digging through the security policy, the architecture
doc and last year's audit report, copy-pasting answers. Then legal asks *"where did
this answer come from?"* — and they dig through it all again.

---

## The solution, by example

You upload two things:

```
Your documents  →  security-policy.pdf, architecture.docx, pentest-report.pdf
The questions   →  client-security-questionnaire.pdf
```

Thirty seconds later:

| # | Question | Verdict | Answer |
|---|---|---|---|
| 1 | Is customer data encrypted at rest? | ✅ **Covered** | Yes — AES-256 on all database volumes.<br>*security-policy.pdf, p.12* |
| 2 | What is your disaster-recovery RTO? | ✅ **Covered** | 4 hours.<br>*architecture.docx, p.5* |
| 3 | Is penetration testing done annually? | ⚠️ **Partial** | A test ran in March 2026, but no annual cadence is stated.<br>*pentest-report.pdf, p.1* |
| 4 | Do you carry cyber liability insurance? | ❌ **Gap** | Not covered by your documents. |

**Result:** 34 answered with sources, 6 flagged for a human. Two days becomes twenty
minutes — and the 6 that need a person are identified *upfront* rather than discovered
at the end.

---

## Why the 6 matter more than the 34

> Any AI tool can produce 40 confident answers. The problem is you don't know which
> ones are made up.
>
> This one tells you it can only answer 34 — and proves each one by quoting your own
> document. **The 6 it refuses are what make the 34 trustworthy.**

In a compliance document, a wrong answer is worse than a blank one. A blank gets filled
in. A confident wrong answer gets signed.

---

## How it avoids making things up

Most AI tools are *asked* not to hallucinate. This one is **checked**.

1. The assistant may only read passages retrieved from your documents.
2. It must quote the exact sentence it relied on.
3. **The app then searches your document for that quote.** If it isn't there — invented
   or even just reworded — the answer is thrown away and marked as a gap.

So a ✅ never means *"the AI said so."* It means *"the sentence was found in your
document."*

---

## Where else it fits

Same tool, different documents — nothing in the code changes:

- **RFP responses** — different questions, identical job
- **Audit prep** — an ISO 27001 or SOC 2 checklist against your real policies, to find
  gaps *before* the auditor does
- **Insurance and claims forms** against a policy document
- **Due diligence** during an acquisition

---

## In one sentence

> It answers a questionnaire from your own documents, cites the exact sentence for
> every answer, and flags the questions your documents can't answer.

---

## Try it in 30 seconds

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

Open **Compliance Copilot**, click **Try it with sample documents** → **Audit 12 questions**.

Then ask the chat *"What is the capital of France?"* — it will refuse, because the
answer isn't in your documents. That refusal is the whole product.

*Setup instructions: [README.md](../README.md)*
