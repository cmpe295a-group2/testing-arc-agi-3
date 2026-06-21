# ARC Prize 2026 — ARC-AGI-3 · Official Kaggle Competition Rules

**Saved 2026-06-20.** Source: competition Rules tab, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 — full verbatim text at the bottom. The section below is OUR strategic reading, not part of the official rules.

---

## ⭐ Strategic implications (what the Rules change for us)

1. **Public vs Private test set — HIDDEN split [§3.18a, §3.7a, §3.18e–f].** "The Competition Data will contain private and public test sets. **Which data belongs to which set will not be made available to Participants.**" The visible leaderboard (Tufa Labs 1.21, pack ~0.65) is the **PUBLIC** score; **prizes are decided on the hidden PRIVATE set.** → You cannot tell which games count. Overfitting the public score (memorize/replay) is the **StochasticGoose trap** (12.58%→0.25% collapse). This validates the whole "build for generalization, not public coverage" thesis — and means `chat_gpt_5_5/`'s public replay would look great on the public leaderboard yet score ~0 on private.

2. **Prizes [§1.5].** $850k total. **$700k bonus needs 100% accuracy** (aspirational — current best is 1.21, i.e. ~1%). The realistic target is the **Top-score track $150k**:
   - Final leaderboard: $40k/$15k/$10k/$5k/$5k
   - **Milestone 1 prizes判定 on June 30, 2026** — notebook must be PUBLISHED by then. *(≈10 days from now.)*
   - Milestone 2: Sept 30, 2026.

3. **1 submission per day [§2.2a].** Slow iteration; up to 2 final submissions. Be deliberate — no brute-forcing the leaderboard.

4. **Open-weight LLM is ALLOWED and a valid path [§2.6b, §2.5a, §2.8].** External models are acceptable if "reasonably accessible to all" at "minimal cost." Winners must open-source **system + model + weights** (OSI / Open Source AI definition). → A proprietary/API model **cannot win** (also no internet); an **open-weight code model (e.g. Qwen-Coder) is explicitly permitted** and satisfies the winner license. Green light for the local-LLM world-model path.

5. **Anti-cheating clause = real DQ risk for the deepcopy/source exploit [§3.8d, §3.16].** Sponsor may disqualify "cheating, deception, or other unfair playing practices," and forbids attempts to "undermine the legitimate operation of the Competition." The games ARE provided as Competition Data (§3.18a includes "executable code provided"), so reading them isn't unauthorized — but exploiting in-process determinism to build a *perfect simulator* that defeats the benchmark's goal-inference intent is a judgment call the Sponsor could rule "unfair." **This is one more reason to treat deepcopy/source as opportunistic insurance, not the bet.**

6. **Winner must deliver reproducible code + do a sponsor interview [§2.8].** Methodology must be documented well enough to reproduce. Favors a clean, explainable approach (good for a 295A writeup too).

## ❓ Still NOT answered by the legal Rules (need the Data + Code Requirements tabs)

The pasted text is the **legal** rules. It does NOT state:
- **How many private games**, and **how the games are delivered** to the scored notebook (mounted `.py` files → OFFLINE mode, source readable → exploit viable; or via a server → COMPETITION/frame-only). → **Data tab.**
- **Notebook runtime: internet ON/OFF at evaluation, GPU type, time limit.** → **Code Requirements** (Overview/Code tab). (§3.15 "Internet" here is only a liability disclaimer, not a runtime statement.)

These two tabs resolve the last decisive unknown (grading mode / frame-only vs source-readable). **Please grab the Data tab and the Code Requirements next.**

---

## Full official text (verbatim, as provided)

# Competition Rules
ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES. See Section 3.18 for defined terms

The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants with the best scores, based on the merits of the data science models submitted. See below for the complete Competition Rules. For Hackathon Competitions, your Submissions will be judged by the Competition Sponsor based on the evaluation rubric set forth on the Competition Website ("Evaluation Rubric"). The Prizes, if any, will be awarded to Participants with the highest ranking(s) as determined by the Competition Sponsor based on the Evaluation Rubric.

You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.

## 1. COMPETITION-SPECIFIC TERMS
1. COMPETITION TITLE — ARC Prize 2026 - ARC-AGI-3
2. COMPETITION SPONSOR — ARC Prize Foundation
3. COMPETITION SPONSOR ADDRESS — 548 Market Street #83849 San Francisco CA 94104
4. COMPETITION WEBSITE — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
5. TOTAL PRIZES AVAILABLE: $850,000
   a. Top score track: $150,000 — b. Bonus Prize: $700,000
   (a) Top score track: $150,000
     (a.1.) Final leaderboard prizes: $75,000 (based on leaderboard score at end of competition)
        First $40,000 · Second $15,000 · Third $10,000 · Fourth $5,000 · Fifth $5,000
     (a.2.) Milestone prizes: $75,000 (based on leaderboard score on two specific dates; Notebooks must be published by the milestone date)
        Milestone 1: June 30th, 2026 — First $25,000 · Second $7,500 · Third $5,000
        Milestone 2: September 30, 2026 — 1st $25,000 · 2nd $7,500 · 3rd $5,000
   (b) Bonus Prize — A Grand Prize of an additional $700,000 unlocked if a team achieves 100% accuracy on the competition leaderboard. Divided among Top 5 teams achieving 100%: First $350,000 · Second $175,000 · Third $70,000 · Fourth $70,000 · Fifth $35,000
6. WINNER LICENSE TYPE — CC-BY 4.0
7. DATA ACCESS AND USE — Apache 2.0

## 2. COMPETITION-SPECIFIC RULES
1. TEAM LIMITS — a. Max team size 8. b. Team mergers allowed (combined submission count ≤ max allowed as of Team Merger Deadline).
2. SUBMISSION LIMITS — a. Max 1 Submission per day. b. Up to 2 Final Submissions for judging. c. Hackathons: 1 only.
3. COMPETITION TIMELINE — Dates on Overview > Timeline page.
4. COMPETITION DATA — a. Data Access and Use: may use Competition Data for any purpose (commercial or non-commercial), incl. participating, forums, academic research/education. Sponsor may disqualify misuse. b. Data Security: use reasonable measures to prevent non-Participants from accessing Competition Data; do not transmit/duplicate/publish/redistribute it; notify Kaggle of unauthorized access.
5. WINNER LICENSE — a. Open Source: license winning Submission + source code under CC-BY 4.0 (OSI-approved, no commercial-use limit). Submissions required to have open source system, open source model, and open source weights/parameters (per Open Source AI definition / OSI checklist). For generally commercially available software not owned by you but procurable without undue expense, you need not grant the license. Incompatible-license input data/pretrained models used to generate the winning solution need not be open-sourced. b. May be required to provide a detailed reproducible description (methodology, architecture, preprocessing, loss, training, hyper-params) + code repo; may be asked to discuss results in a recorded/panel call.
6. EXTERNAL DATA AND TOOLS — a. May use External Data if publicly available + equally accessible to all at no cost, or meets Reasonableness criteria (2.6.b). b. External data and models acceptable unless specifically prohibited by Host; must be "reasonably accessible to all" and "minimal cost"; Host applies a "Reasonableness Standard." Example: a small subscription (e.g. Gemini Advanced) is acceptable; a proprietary dataset license exceeding a prize's cost is not. c. AMLT (automated ML tools) allowed if appropriately licensed.
7. ELIGIBILITY — a. Competition Entity employees etc. may participate but not win prizes.
8. WINNER'S OBLIGATIONS — a. Deliver final model's software code (training + inference) + docs + description of required compute environment, capable of generating the winning Submission. a. Conduct an interview with the sponsor + work with a technical writer to document the solution. b. For commercial off-the-shelf software not owned by you but procurable without undue expense, identify it instead of delivering code. c. AMLT submissions may win but must meet all Rules. Grant Sponsor the winning-Submission license; sign/return all prize acceptance + tax docs.
9. GOVERNING LAW — California law, Santa Clara County courts.

## 3. GENERAL COMPETITION RULES — BINDING AGREEMENT
1. ELIGIBILITY — registered Kaggle account; ≥18 / age of majority; not resident of Crimea, DNR, LNR, Cuba, Iran, North Korea; not under U.S. export controls/sanctions. Open worldwide with those exceptions. Entity/employer representation binds you and the entity. Sponsor may verify eligibility; false info → disqualification.
2. SPONSOR AND HOSTING PLATFORM — Hosted by Kaggle on behalf of Sponsor; Kaggle is an independent contractor, not a party to your agreement with Sponsor; handles administration. You are subject to Kaggle Terms of Service.
3. COMPETITION PERIOD — Runs Start Date → Final Submission Deadline. Timeline subject to change; check Website. You determine your time zone.
4. COMPETITION ENTRY — a. No purchase necessary; register before Entry Deadline; follow Requirements; meet deadlines. b. Submissions may not use hand labeling / human prediction of validation/test data (except as allowed in Hackathons). c. Multi-stage competitions may require valid Submissions each stage. d. Submissions void if illegible/incomplete/damaged/altered/counterfeit/fraudulent/late. Sponsor may disqualify rule-breakers.
5. INDIVIDUALS AND TEAMS — a. One unique account only; multi-account → disqualification. b. Teams: join/form only one Team; each member a separate account; confirm membership; ≤ Max Team Size. c. Team Merger conditions (size, submission counts, deadlines). d. **Private Sharing: No private sharing outside of Teams.** Sharing code is OK only if made available to all Participants on the forums.
6. SUBMISSION CODE REQUIREMENTS — a. **Private Code Sharing prohibited** during the Competition Period (sharing Competition Code between separate Teams → possible disqualification). b. **Public Code Sharing permitted** only on Kaggle forums/notebooks for the Competition; deemed licensed under an OSI-approved license (no commercial-use limit). c. Use of Open Source: any open-source code used must be under an OSI-approved license with no commercial-use limit.
7. DETERMINING WINNERS — a. Scored/ranked by the evaluation metric on the Website. **Public Leaderboard = public test set; Private Leaderboard = private test set. Potential winners determined solely by the Private Leaderboard.** b. Ties broken by earliest Submission; disqualified winner → next rank.
8. NOTIFICATION OF WINNERS & DISQUALIFICATION — a. Notified by email. b. Non-response within 1 week or opt-out → no prize, alternate selected. c./d. Sponsor may disqualify for cheating/deception/unfair playing practices/harassment. e. Disqualified Participants may be removed from leaderboard (and lose points/medals). f. Final leaderboard publicly displayed; Sponsor determinations final.
9. PRIZES — Subject to review/verification of eligibility + compliance. Non-compliance → disqualify Submission or require remediation within 1 week. Decline allowed. Return acceptance docs within 2 weeks; prizes within ~30 days. No transfer/assignment. Team prizes split evenly unless unanimous otherwise.
10. TAXES — Winners responsible for all taxes; must submit tax docs; U.S. residents get IRS Form-1099.
11. GENERAL CONDITIONS — All applicable laws/regulations apply.
12. PUBLICITY — Sponsor/Kaggle may use your name/likeness for promotion (unless prohibited by law).
13. PRIVACY — Personal Information collected/stored/shared per Kaggle Privacy Policy; transferred to Sponsor (independent controller); may be outside your country.
14. WARRANTY, INDEMNITY AND RELEASE — Submission is your original work; no infringement; you indemnify Competition Entities; release them from liability for Website malfunctions / Submission processing errors.
15. INTERNET — Competition Entities not responsible for technical failures, lost/late Submissions, network issues, etc. (liability disclaimer only).
16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY — Sponsor may cancel/terminate/modify/suspend if the Competition is compromised (virus, bugs, tampering, fraud, technical failure). May disqualify tampering; deliberate damage/undermining is a violation of criminal/civil law.
17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT.
18. DEFINITIONS — a. **"Competition Data"** = data/datasets from the Website for the Competition, **including any prototype or executable code provided.** Contains private and public test sets; **which data belongs to which set will not be made available to Participants.** b. "Entry" = joined/accepted rules. c. "Final Submission" = used for final leaderboard placement. d. "Participant" = an entrant who makes a Submission. e. **"Private Leaderboard"** = scores vs the private test set; determines final standing. f. **"Public Leaderboard"** = scores vs a representative sample of the test data; visible throughout. g. "Sponsor" hosts the competition. h. "Submission" = model/notebook/prediction file/etc. provided for evaluation. i. "Team" = one or more Participants officially merged.
