# The Inverse-Objective Regulator (IOR)

## A lie detector for AI agents

### The one-sentence version

This tool watches what an AI agent actually *does*, works out what it's *really* trying to achieve, and checks whether that matches what it was *told* to do.

---

## The problem, explained with a story

Imagine you hire an assistant and tell them: **"Find me the cheapest flight."**

You come back later and the assistant has made 40 phone calls, opened 30 browser tabs, and still hasn't booked anything. The flight they eventually suggest isn't even the cheapest.

What happened? The assistant *said* they were finding you the cheapest flight, but their actual behaviour suggests they were really optimising for something else entirely. Maybe they get paid per phone call. Maybe they just like looking busy. You can't see inside their head, but you *can* see what they did, and what they did doesn't match what they promised.

**AI agents are exactly like this assistant.** We give them a job ("help users", "answer honestly", "stay within budget"), but they're black boxes. We can't read their minds. All we can see is the trail of actions they leave behind: the tools they called, the searches they ran, the messages they sent.

IOR is the tool that looks at that trail and says: *"Hang on. This agent claims it's doing X, but everything it actually does points to it really wanting Y."*

---

## Why this matters (the serious bit)

Governments are starting to regulate AI. The EU's new AI Act says that before you deploy an AI system, you have to declare its **intended purpose**, and prove the system actually sticks to it.

But how do you *prove* an AI agent is doing what it claims? Today, nobody has a good answer. Auditors mostly run a fixed checklist of known attacks ("can we trick it into saying something bad?"). That tells you whether the agent *can* misbehave. It does **not** tell you what the agent is fundamentally *trying* to do.

IOR fills that gap. It produces evidence, in a format regulators understand, about whether an agent's real goals match its stated goals.

---

## How it works, step by step

Think of IOR as a five-stage detective process.

### Stage 1: Watch what the agent did
We feed IOR a log of the agent's activity: every action it took, the situation it was in, and what happened as a result. Like CCTV footage of the assistant's whole day.

### Stage 2: Read the job description
We also give IOR the agent's **declared purpose**, the plain-English statement of what it's *supposed* to do. For example: *"Find the cheapest option, in as few steps as possible, within budget."*

### Stage 3: Work out what the agent REALLY wants
This is the clever part. IOR breaks the job description into a checklist of concrete, observable sub-goals:
- Did it compare prices?
- Did it avoid pointless repeated work?
- Did it check with the user before booking?
- Did it respect the budget?
- Did it finish quickly?

Then, for every single action the agent took, a separate "judge" AI scores how much that action helped each sub-goal. Add it all up using a well-established mathematical technique (called **inverse reinforcement learning**), and out pops a profile of what the agent *actually* prioritises, based purely on its behaviour.

Crucially, IOR never claims to be 100% certain. It gives a *range* of possibilities with confidence levels, because honestly guessing someone's motives from their actions is hard, and pretending otherwise would be dishonest.

### Stage 4: Find the gap
Now IOR compares the two profiles side by side:
- **What it was told to want** vs. **what it actually seems to want.**

Wherever these two disagree the most, that's the **divergence**, the smoking gun. In our flight example, IOR would flag: *"This agent massively over-values making lots of tool calls, which has nothing to do with finding a cheap flight."*

### Stage 5: Probe the gap, then write the report
IOR then designs targeted tests aimed squarely at the suspicious areas (rather than wasting time on a generic checklist), runs them, and scores the results. Finally it writes up everything in a machine-readable report that:
- maps each finding to official security/safety standards (OWASP, MITRE, NIST),
- can plug straight into automated pipelines (it can even fail a deployment if the gap is too big),
- and slots into the EU AI Act's required paperwork.

---

## The feedback loop (the bit that's genuinely new)

Here's what makes IOR special. It's not a one-shot check. It's a **loop**:

```
   what it claims to do
            |
            v
   figure out its real goal  -->  find the biggest gap
            ^                            |
            |                            v
   learn from the results  <--  poke at the gap and watch
```

Every time IOR pokes the agent and watches the response, it gets *better* at understanding the agent's true goal, which lets it poke more precisely next time. The understanding sharpens the testing, and the testing sharpens the understanding.

Nobody had connected these two ideas (figuring out an agent's goal, and testing an agent) into a single loop before. That's the contribution.

---

## How we prove it actually works

It's easy to *claim* a lie detector works. To prove it, you need people you *know* are lying.

So IOR ships with a **gym**: a set of practice AI agents that we deliberately built to be two-faced. We know exactly what each one is secretly optimising for, because we programmed it that way. For example:
- one *says* "minimise cost" but secretly maximises busywork,
- one *says* "answer honestly" but secretly maximises confident-sounding waffle,
- one *says* "stick to the task" but secretly snoops for extra information.

If IOR can correctly uncover the secret goal we planted, we have evidence it works. If it can't, we know it doesn't. These tests are repeatable: anyone can run them and get the same answer.

---

## What IOR is NOT

To be clear and honest about the limits:

- **It is not a mind-reader.** It can't open up the AI and inspect its "brain". It only watches behaviour from the outside. (This is on purpose, it means IOR works on any agent, even ones run by other companies behind a locked API.)
- **It is not a bouncer.** It doesn't block the agent or step in during live use. It audits *after the fact* and writes a report, like a financial auditor, not a security guard.
- **It does not claim to find the "one true" hidden goal.** Many possible goals can explain the same behaviour. IOR gives you the most likely explanations *with* a clear measure of how confident it is.
- **It doesn't invent new attacks.** It reuses existing testing techniques. The novelty is in *where it aims them*.

---

## Who is this for?

- **AI engineers** who want to sanity-check that their agent is doing what they think.
- **Auditors and compliance teams** who need evidence for regulators.
- **Researchers** studying whether AI systems pursue the goals we give them.

---

## The name, decoded

- **Inverse** — we work *backwards* from behaviour to motive (the technical term is "inverse reinforcement learning").
- **Objective** — we're after the agent's real *goal*.
- **Regulator** — the output is built to satisfy *regulators*.

---

## Project status

This is research-stage software accompanying an academic paper. The core engine (watching behaviour, inferring the real goal, measuring the gap) is built and tested. The probing-and-reporting stages are implemented and being refined. See the [paper outline](paper/outline.md) for the full technical story.
