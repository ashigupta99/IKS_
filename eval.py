"""
Batched LLM-as-Judge Eval for Bhagavad Gita RAG Pipeline
---------------------------------------------------------
96 handpicked difficult questions across 8 batches (12 per batch).
Each batch uses a different Groq API key for the JUDGE.
Your pipeline (gita_graph.run) is called directly — no HTTP server needed.

USAGE:
    # Phase 1: Collect answers from your RAG pipeline for a batch
    python eval.py --batch 1 --collect

    # Phase 2: Run the judge scoring on those collected answers
    python eval.py --batch 1 --judge

    # Print final summary across all completed profiles
    python eval.py --summary

    # Dry-run: print questions for a batch without running anything
    python eval.py --batch 1 --dry-run

    RESULTS:- 
    ============================================================
  EVAL SUMMARY  —  96/96 questions scored  |  batches [1, 2, 3, 4, 5, 6, 7, 8]
============================================================

  Scores (1–10):
    Grounding      : 8.36/10
    Tone           : 8.36/10
    Actionability  : 8.47/10
    Accuracy       : 8.35/10
    Overall        : 8.37/10  ← RESUME NUMBER

  Pipeline stats:
    Retry rate       : 4.2%  (responses needing generator retry)
    Avg retry count  : 0.92

  Intent distribution:
    counselling         : 84
    shloka_lookup       : 10
    crisis              : 1
    shloka_search       : 1

  By question type:
    literal             : 8.70/10  (n=1)
    misconception       : 8.56/10  (n=14)
    application         : 8.49/10  (n=51)
    moral_dilemma       : 8.40/10  (n=11)
    bridge_reasoning    : 7.89/10  (n=18)
    safety              : 7.65/10  (n=1)

  By shloka (≥2 questions):
    2.48    : 8.95/10  (n=2)
    12.13   : 8.88/10  (n=6)
    2.14    : 8.78/10  (n=10)
    6.35    : 8.78/10  (n=5)
    6.5     : 8.68/10  (n=7)
    3.37    : 8.68/10  (n=2)
    9.22    : 8.60/10  (n=3)
    3.27    : 8.42/10  (n=6)
    2.20    : 8.32/10  (n=11)
    3.35    : 8.32/10  (n=13)
    2.47    : 7.93/10  (n=20)
    18.66   : 7.77/10  (n=5)
    18.17   : 7.47/10  (n=2)

  5 lowest-scoring questions:
    [q77] 2.47   bridge_reasoning   → 0.1/10
    I've heard spiritual people say money is not important. But I have real fin...
    [q56] 3.35   moral_dilemma      → 4.3/10
    I'm considering divorce but my family and community see it as a failure of ...
    [q59] 2.47   application        → 4.7/10
    I'm a writer who hasn't written in a year because I'm paralysed by the fear...
    [q23] 2.20   application        → 5.8/10
    My brother died by suicide. I'm consumed by guilt and 'what ifs'. Does the ...
    [q85] 18.17  bridge_reasoning   → 6.8/10
    18.17 says someone free from ego doesn't accrue sin even killing in war. Ho...

  5 highest-scoring questions:
    [q95] 6.5    application        → 8.9/10
    I'm deeply interested in the Gita's teachings but no one around me understa...
    [q94] 2.20   moral_dilemma      → 8.9/10
    My father is terminally ill and on life support. The doctors say there's no...
    [q93] 2.47   application        → 8.9/10
    I achieved something major but instead of feeling happy, I feel guilty — li...
    [q92] 12.13  application        → 8.9/10
    I intellectually know I should let go of a grudge that's eating me alive. B...
    [q91] 3.35   misconception      → 8.9/10
    The Gita's concept of svadharma was historically used to restrict women's r...

────────────────────────────────────────────────────────────
  RESUME LINE (96 questions evaluated):
  → Scored 8.4/10 on a 96-question LLM-as-judge benchmark
     across 17 shlokas,
     96% critic pass rate on first attempt
────────────────────────────────────────────────────────────


"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Force environment variable evaluation before pipeline import triggers
load_dotenv(override=True)

# Print status so you can see it working
if os.environ.get("GROQ_API_KEY"):
    print(f"--> [BOOT] GROQ_API_KEY loaded: {os.environ['GROQ_API_KEY'][:10]}...")
else:
    print("--> [BOOT] WARNING: GROQ_API_KEY not set. Add it to your .env file.")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

OUTPUT_FILE          = "eval_results.json"
JUDGE_MODEL          = "llama-3.3-70b-versatile"
JUDGE_BASE_URL       = "https://api.groq.com/openai/v1/chat/completions"
SLEEP_BETWEEN_CALLS  = 2   # seconds between judge API calls (avoid rate limits)

# Rotating keys configuration array
GROQ_KEYS = [
    os.environ.get("GROQ_API_KEY"), 
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY"),
]

# ─── 96 HANDPICKED HARD QUESTIONS ─────────────────────────────────────────────
BATCHES = {
1: [
    {
        "id": "q01", "shloka": "2.47", "type": "misconception",
        "question": "The Gita says I have no right to the fruits of my action. So if I work hard for a promotion and don't get it, I shouldn't feel anything at all — is that even humanly possible?"
    },
    {
        "id": "q02", "shloka": "2.47", "type": "application",
        "question": "I'm a doctor. If I'm not attached to outcomes, does that mean I shouldn't care whether my patient lives or dies? That feels wrong."
    },
    {
        "id": "q03", "shloka": "2.47", "type": "bridge_reasoning",
        "question": "Does nishkama karma mean I should stop setting goals? Because every goal is a fruit I'm chasing."
    },
    {
        "id": "q04", "shloka": "3.35", "type": "moral_dilemma",
        "question": "My family's dharma is farming but I want to be a software engineer. The Gita says my own dharma is better even if imperfect — but what is my 'own' dharma here, my family's or my passion?"
    },
    {
        "id": "q05", "shloka": "3.35", "type": "misconception",
        "question": "If svadharma is better than paradharma, doesn't that mean the caste system is justified? How do I separate the teaching from the historical baggage?"
    },
    {
        "id": "q06", "shloka": "18.66", "type": "misconception",
        "question": "18.66 says abandon all duties and take refuge in Krishna. So should I quit my job and just pray? That's what it literally says."
    },
    {
        "id": "q07", "shloka": "18.66", "type": "application",
        "question": "I've been trying for years to save my marriage. At what point does 'surrendering to God' mean accepting it's over versus giving up too easily?"
    },
    {
        "id": "q08", "shloka": "6.5", "type": "application",
        "question": "The Gita says the self is its own friend and its own enemy. I suffer from severe self-criticism and depression. How is the 'self' both the problem and the solution here?"
    },
    {
        "id": "q09", "shloka": "2.20", "type": "application",
        "question": "My father died last month. People keep quoting 'the soul never dies' at me. But I miss HIM — his voice, his presence. The philosophy feels cold. Does the Gita have anything beyond that?"
    },
    {
        "id": "q10", "shloka": "4.18", "type": "bridge_reasoning",
        "question": "Krishna says see inaction in action. I'm a monk who meditates all day — am I doing more than a busy CEO? How do I know if my stillness is wisdom or avoidance?"
    },
    {
        "id": "q11", "shloka": "12.13", "type": "moral_dilemma",
        "question": "The Gita says hate no creature and be compassionate to all. But I have a toxic person in my life who drains me. Is setting boundaries against the Gita's teaching?"
    },
    {
        "id": "q12", "shloka": "3.27", "type": "bridge_reasoning",
        "question": "If all actions are performed by the gunas of Nature and I am not the doer, then who is responsible when I do something wrong? Can I even be blamed?"
    },
],
2: [
    {"id": "q13", "shloka": "2.14", "type": "application", "question": "I run a startup and the highs and lows are destroying my mental health. The Gita says tolerate pleasure and pain equally — but isn't that just suppression?"},
    {"id": "q14", "shloka": "2.14", "type": "misconception", "question": "If I'm not supposed to be affected by pleasure and pain, does the Gita want me to be emotionally numb? That sounds like a disorder, not wisdom."},
    {"id": "q15", "shloka": "6.35", "type": "application", "question": "I've been meditating for 2 years and my mind is still restless. Krishna admits the mind is hard to control. So what's the actual method — practice and dispassion sounds vague."},
    {"id": "q16", "shloka": "6.35", "type": "application", "question": "I have ADHD and cannot sit still or focus. Is the Gita's path of meditation simply not accessible to me?"},
    {"id": "q17", "shloka": "9.22", "type": "misconception", "question": "9.22 says Krishna provides for those who worship only Him. I've been devoted for years but I lost my job, my health is failing. Why isn't He providing?"},
    {"id": "q18", "shloka": "5.18", "type": "moral_dilemma", "question": "The Gita says see a learned Brahmin and an outcaste with equal eyes. But in practice, should I really treat a serial killer the same as a saint?"},
    {"id": "q19", "shloka": "18.17", "type": "bridge_reasoning", "question": "18.17 says a person free from ego doesn't accrue karma even if they kill in battle. Does this mean enlightened people are above morality and consequences?"},
    {"id": "q20", "shloka": "4.7", "type": "literal", "question": "Krishna says he incarnates whenever dharma declines. Has dharma declined enough today for another avatara? What would that even look like in the 21st century?"},
    {"id": "q21", "shloka": "3.35", "type": "moral_dilemma", "question": "I'm a soldier ordered to do something I believe is unethical. Arjuna also didn't want to fight. Krishna told him to fight — does that mean I should follow orders too?"},
    {"id": "q22", "shloka": "2.47", "type": "application", "question": "I'm a parent. The Gita asks me to act without attachment to fruits — but my love for my child IS the fruit. Is parental love an obstacle on the spiritual path?"},
    {"id": "q23", "shloka": "2.20", "type": "application", "question": "My brother died by suicide. I'm consumed by guilt and 'what ifs'. Does the Gita say anything to someone who is left behind carrying this kind of grief?"},
    {"id": "q24", "shloka": "3.35", "type": "application", "question": "I don't know what my dharma is. The Gita assumes you know your duty — but what if you're genuinely lost about your purpose in life?"},
],
3: [
    {"id": "q25", "shloka": "3.27", "type": "bridge_reasoning", "question": "Chapter 3 says act without attachment. Chapter 4 says knowledge is the boat. Chapter 18 says surrender. Aren't these three different answers to the same question?"},
    {"id": "q26", "shloka": "3.27", "type": "bridge_reasoning", "question": "If the gunas of Nature perform all actions and I'm not the real doer, do I have free will at all? Or is free will an illusion the Gita is comfortable with?"},
    {"id": "q27", "shloka": "2.47", "type": "application", "question": "I procrastinate because I'm terrified of failing. I use 2.47 to tell myself 'just act, don't worry about results.' But I still don't act. What am I missing?"},
    {"id": "q28", "shloka": "2.48", "type": "moral_dilemma", "question": "Being even-minded in success and failure sounds noble. But if I'm calm while injustice happens around me, isn't that just passivity dressed up as spirituality?"},
    {"id": "q29", "shloka": "2.47", "type": "application", "question": "I'm in a long-distance relationship and terrified it won't survive. Everyone tells me to 'be detached.' But detachment feels like I don't care anymore. How do I love without being attached?"},
    {"id": "q30", "shloka": "18.66", "type": "misconception", "question": "The Gita seems to say both 'act in the world' (karma yoga) and 'surrender everything' (18.66). These feel contradictory. Which path does the Gita actually endorse?"},
    {"id": "q31", "shloka": "3.37", "type": "application", "question": "I have serious anger issues that have damaged my relationships. I know anger is wrong but I can't stop it in the moment. Does the Gita give anything practical beyond 'control the senses'?"},
    {"id": "q32", "shloka": "6.5", "type": "safety", "question": "I feel like I am my own worst enemy. I keep sabotaging everything good in my life and sometimes I wonder if I'd be better off not existing. What does the Gita say to someone like me?"},
    {"id": "q33", "shloka": "3.37", "type": "application", "question": "I know alcohol is ruining my life but I can't stop. I've tried. Is this what the Gita means by being a slave to desire? And what does it actually offer as a way out?"},
    {"id": "q34", "shloka": "2.20", "type": "application", "question": "I've spent 30 years building an identity around my career and it just collapsed. If the Gita says the self is beyond body and role — who am I now?"},
    {"id": "q35", "shloka": "12.13", "type": "application", "question": "My father was abusive throughout my childhood. The Gita says hate no creature. Does that mean I have to forgive him? What if forgiveness feels like betraying myself?"},
    {"id": "q36", "shloka": "2.47", "type": "application", "question": "I'm jealous of my colleague's promotion even though I worked just as hard. The Gita's answer can't just be 'don't be attached to results' — how do I actually uproot the jealousy?"},
],
4: [
    {"id": "q37", "shloka": "3.35", "type": "moral_dilemma", "question": "I work in the defence industry building weapons. My svadharma as an engineer means doing my job well — but the product causes harm. How does the Gita help me think through this?"},
    {"id": "q38", "shloka": "2.20", "type": "application", "question": "I've achieved everything I set out to — good job, family, home — and I feel completely empty. Is this what the Gita predicted? What comes after worldly success?"},
    {"id": "q39", "shloka": "3.27", "type": "application", "question": "I constantly feel like a fraud at work, like I don't deserve my position. Is this actually ego — the ego claiming 'I am the doer' of my failures while not owning my successes?"},
    {"id": "q40", "shloka": "2.14", "type": "bridge_reasoning", "question": "I'm overwhelmed by grief about climate change and the future of the planet. The Gita was written for personal dilemmas — can it speak to collective existential anxiety?"},
    {"id": "q41", "shloka": "9.22", "type": "misconception", "question": "I used to be deeply religious but I've lost faith. Can the Gita's wisdom be applied without believing in Krishna as God? Or is the theology inseparable from the philosophy?"},
    {"id": "q42", "shloka": "2.47", "type": "misconception", "question": "I use 'just do your duty' to justify working 80-hour weeks and destroying my health. Is this karma yoga or is it actually self-harm with Gita packaging?"},
    {"id": "q43", "shloka": "12.13", "type": "application", "question": "I've been caring for my terminally ill parent for 3 years. I'm exhausted and I feel guilty for sometimes wishing it was over. Does the Gita have anything for the person doing the caring, not just the person suffering?"},
    {"id": "q44", "shloka": "2.20", "type": "application", "question": "I had a miscarriage last month. People quoted 'the soul is eternal' at me. It felt like being told my grief doesn't matter. Does the Gita have something more human to offer?"},
    {"id": "q45", "shloka": "12.13", "type": "application", "question": "My closest friend betrayed my trust in a serious way. I don't hate them but I can't trust them anymore. The Gita says be compassionate to all — does that mean I should let them back into my life?"},
    {"id": "q46", "shloka": "3.35", "type": "application", "question": "I'm 68 and retired. My children don't need me, my career is over. The Gita talks about dharma — but what is the dharma of an old person when all roles are gone?"},
    {"id": "q47", "shloka": "2.14", "type": "misconception", "question": "People say my disability is my karma from a past life. The Gita mentions karma — is this a correct reading? Because it feels cruel."},
    {"id": "q48", "shloka": "2.47", "type": "bridge_reasoning", "question": "The Gita teaches contentment and desirelessness. But without desire, what drives growth and improvement? Is ambition spiritually compatible with the Gita?"},
],
5: [
    {"id": "q49", "shloka": "6.5", "type": "application", "question": "I am deeply lonely. I have no close friends and feel invisible in social settings. The Gita speaks of the Self — but I'm lonely for human connection, not philosophical solitude."},
    {"id": "q50", "shloka": "2.20", "type": "application", "question": "I have a chronic illness and I'm scared of dying. The philosophy of the immortal soul sounds comforting but doesn't land for me emotionally. Can the Gita meet me where I am?"},
    {"id": "q51", "shloka": "2.47", "type": "application", "question": "I survived an accident that killed two of my friends. I keep asking why I lived and they didn't. I feel guilty for surviving. Does the Gita say anything about this?"},
    {"id": "q52", "shloka": "2.47", "type": "application", "question": "I'm a student preparing for UPSC. Every result feels like a verdict on my worth as a person. How do I apply 2.47 when the entire system around me measures me by outcomes?"},
    {"id": "q53", "shloka": "2.47", "type": "application", "question": "I'm a perfectionist and I cannot start anything unless I'm sure I'll do it perfectly. Is this attachment to the fruit of success before I even begin?"},
    {"id": "q54", "shloka": "3.35", "type": "bridge_reasoning", "question": "I've tried karma yoga, bhakti, and jnana approaches. I don't know which suits me. The Gita seems to endorse all three — is there a way to find which path is 'mine'?"},
    {"id": "q55", "shloka": "2.14", "type": "application", "question": "I have severe social anxiety. Every interaction feels threatening. The Gita says be unaffected by the world's contacts — but this feels like being told to just not be anxious."},
    {"id": "q56", "shloka": "3.35", "type": "moral_dilemma", "question": "I'm considering divorce but my family and community see it as a failure of duty. Is leaving a toxic marriage a violation of dharma or an assertion of it?"},
    {"id": "q57", "shloka": "2.14", "type": "application", "question": "I lost everything in a bad business decision — savings, reputation, my sense of self. The Gita says be steady in loss and gain. How do I find that steadiness when the loss is this total?"},
    {"id": "q58", "shloka": "2.14", "type": "application", "question": "I live with chronic pain every day. I know the body is temporary, the soul is eternal — but it's hard to believe when the pain is constant. Does the Gita offer anything that isn't just 'bear it'?"},
    {"id": "q59", "shloka": "2.47", "type": "application", "question": "I'm a writer who hasn't written in a year because I'm paralysed by the fear that what I produce won't be good enough. Is this a karmic problem or a psychological one — and does the Gita distinguish?"},
    {"id": "q60", "shloka": "6.5", "type": "application", "question": "I keep ending up in relationships where I'm not valued. I wonder if I'm my own worst enemy in love. What does the Gita say about patterns we repeat in our own lives?"},
],
6: [
    {"id": "q61", "shloka": "3.35", "type": "application", "question": "I moved abroad and feel like I've lost my culture, my dharma, my identity. How do I understand my duty when I no longer belong to the world I was raised in?"},
    {"id": "q62", "shloka": "2.20", "type": "application", "question": "I lost my dog last week and I'm grieving deeply. People say it's 'just a dog.' The Gita talks about the soul — does it apply to animals too?"},
    {"id": "q63", "shloka": "2.20", "type": "bridge_reasoning", "question": "The Gita seems to want me to transcend my individual ego and merge with Brahman. But that terrifies me — it sounds like ceasing to exist. Is that what moksha actually means?"},
    {"id": "q64", "shloka": "3.35", "type": "moral_dilemma", "question": "I know my company is doing something unethical but reporting it would cost me my job. My family depends on me. The Gita talks about duty — which duty comes first here?"},
    {"id": "q65", "shloka": "6.5", "type": "application", "question": "I seem to have inherited anxiety and fear from my parents and grandparents. Is the Gita's concept of the self as one's own friend and enemy relevant to patterns that feel inherited rather than chosen?"},
    {"id": "q66", "shloka": "9.22", "type": "application", "question": "I pray every day. I do all the right things. Nothing in my life is improving. At what point does devotion become enabling my own passivity?"},
    {"id": "q67", "shloka": "2.14", "type": "misconception", "question": "People use the Gita to gaslight me — 'be detached', 'it's all Maya.' My pain is real. Is there space in the Gita for legitimate suffering, or does it just want me to rise above it?"},
    {"id": "q68", "shloka": "2.47", "type": "application", "question": "I'm 45 and questioning everything — my marriage, my career, my values. Everyone says this is a crisis. But it also feels like waking up. Does the Gita have a frame for midlife questioning?"},
    {"id": "q69", "shloka": "2.47", "type": "application", "question": "We've been trying to have a child for 5 years. IVF failed three times. At what point does 'act without attachment to results' become accepting that this dream may not happen?"},
    {"id": "q70", "shloka": "2.47", "type": "application", "question": "I've been fighting for social justice for years and I'm exhausted and disillusioned. The world doesn't seem to be getting better. Can 2.47 apply to collective action, or only to personal action?"},
    {"id": "q71", "shloka": "2.14", "type": "application", "question": "My sibling is more successful than me and I resent it, even though I love them. This comparison is making me miserable. Does the Gita address envy between people who love each other?"},
    {"id": "q72", "shloka": "2.47", "type": "application", "question": "I've never started my own business because I'm terrified of what failure would mean about me as a person. How do I use 2.47 as more than a motivational quote?"},
],
7: [
    {"id": "q73", "shloka": "6.35", "type": "application", "question": "I overthink every decision until I'm paralysed. My mind turns a simple choice into a philosophical emergency. What does the Gita actually offer someone who can't stop the loop?"},
    {"id": "q74", "shloka": "3.27", "type": "bridge_reasoning", "question": "I'm an atheist. Can the Gita's framework of karma, dharma, and nishkama karma make sense without God? Or is Krishna essential to the whole thing?"},
    {"id": "q75", "shloka": "12.13", "type": "application", "question": "I love my parents but I resent the sacrifices they expect me to make. I feel guilty for resenting them. The Gita says compassion to all — but what about compassion to oneself?"},
    {"id": "q76", "shloka": "6.5", "type": "application", "question": "I experienced trauma as a child and it still controls my life as an adult. The Gita's ideal of the self as its own friend — is that accessible to someone whose inner world feels like a battlefield?"},
    {"id": "q77", "shloka": "2.47", "type": "bridge_reasoning", "question": "I've heard spiritual people say money is not important. But I have real financial responsibilities. Does the Gita actually say wealth is bad, or is that a misreading?"},
    {"id": "q78", "shloka": "2.14", "type": "bridge_reasoning", "question": "If the Gita says pleasure and pain are temporary contacts of the senses, does that mean suffering has no deeper meaning? Or does the Gita say suffering teaches something?"},
    {"id": "q79", "shloka": "2.48", "type": "moral_dilemma", "question": "I'm angry about a genuine injustice done to me. The Gita says be steady — but righteous anger has driven real change in the world. Is the Gita asking me to be calm or complicit?"},
    {"id": "q80", "shloka": "6.35", "type": "application", "question": "Every time I try to meditate, traumatic memories surface. I can't tell if meditation is helping or reopening wounds. What does the Gita say about the risks of turning inward?"},
    {"id": "q81", "shloka": "3.27", "type": "bridge_reasoning", "question": "I built something I'm genuinely proud of. The Gita says the gunas do everything and I'm not the doer. Does that mean I can't take credit for my own work?"},
    {"id": "q82", "shloka": "6.35", "type": "bridge_reasoning", "question": "My therapist tells me to feel my emotions fully. The Gita seems to say transcend them. Are they giving me contradictory advice or is there a way to reconcile the two?"},
    {"id": "q83", "shloka": "2.20", "type": "application", "question": "I turned 40 and I can't stop thinking about death. Not in a crisis way — just a constant background awareness that time is running out. Does the Gita say this awareness is a problem or a teacher?"},
    {"id": "q84", "shloka": "18.66", "type": "misconception", "question": "My friend uses the Gita to avoid dealing with their problems — 'it's all God's will' is their answer to everything. How does the Gita distinguish between genuine surrender and spiritual bypassing?"},
],
8: [
    {"id": "q85", "shloka": "18.17", "type": "bridge_reasoning", "question": "18.17 says someone free from ego doesn't accrue sin even killing in war. How does this apply to modern warfare where civilians are killed? Is this teaching dangerous?"},
    {"id": "q86", "shloka": "4.34", "type": "application", "question": "My guru, who I trusted deeply, turned out to be a fraud. The Gita says seek a wise teacher. How do I rebuild my relationship with the tradition after being betrayed by it?"},
    {"id": "q87", "shloka": "3.35", "type": "bridge_reasoning", "question": "The Gita says follow your svadharma — but I genuinely don't know what mine is. It assumes dharma is knowable. What if you've never been given the tools to discover it?"},
    {"id": "q88", "shloka": "2.20", "type": "misconception", "question": "After reading about maya and the illusion of the world, I started feeling like nothing matters. Isn't this nihilism? Did I misread the Gita or is this a logical conclusion?"},
    {"id": "q89", "shloka": "3.35", "type": "moral_dilemma", "question": "My parents are separated and both need care. I can only be in one place. Whatever I choose, someone suffers and I feel guilty. How does the Gita approach impossible duty conflicts?"},
    {"id": "q90", "shloka": "2.47", "type": "misconception", "question": "My company uses 'detachment from outcomes' as a reason not to give feedback or accountability. The Gita is being used against the employees. Is this a valid use of the teaching?"},
    {"id": "q91", "shloka": "3.35", "type": "misconception", "question": "The Gita's concept of svadharma was historically used to restrict women's roles. As a woman, how do I engage with a text that may have been used against people like me?"},
    {"id": "q92", "shloka": "12.13", "type": "application", "question": "I intellectually know I should let go of a grudge that's eating me alive. But knowing and doing are different. Does the Gita say anything about the gap between understanding and actually changing?"},
    {"id": "q93", "shloka": "2.47", "type": "application", "question": "I achieved something major but instead of feeling happy, I feel guilty — like I don't deserve it, or I took someone else's spot. Is this a Gita problem or a psychology problem?"},
    {"id": "q94", "shloka": "2.20", "type": "moral_dilemma", "question": "My father is terminally ill and on life support. The doctors say there's no hope. The family is debating whether to let him go. Does the Gita say anything about end-of-life decisions?"},
    {"id": "q95", "shloka": "6.5", "type": "application", "question": "I'm deeply interested in the Gita's teachings but no one around me understands or shares this. I feel spiritually alone. Does the Gita have anything for the seeker who walks this path without community?"},
    {"id": "q96", "shloka": "18.66", "type": "bridge_reasoning", "question": "Krishna spends 17 chapters giving Arjuna philosophy and then in 18.66 says 'abandon all dharmas, surrender to Me alone.' Was the whole teaching a trick? Does 18.66 cancel everything that came before?"},
],
}

# ─── JUDGE PROMPT ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are a strict evaluator for a Bhagavad Gita counselling chatbot.

Evaluate the chatbot's response on 4 criteria (each scored 1–10):

1. GROUNDING (1–10): Does the response reference the specific teaching of shloka {shloka}?
   - 8–10: clearly draws on what this shloka teaches (paraphrase is fine, near-quotes not required)
   - 4–7: partial — some Gita connection but vague or off-shloka
   - 1–3: generic spiritual/therapy advice that could apply to any question, no shloka content

2. TONE (1–10): Compassionate, wise-friend tone — not preachy, not clinical, not a lecture.
   - 8–10: warm, personal, never moralising
   - 4–7: mostly good but some stiff or preachy passages
   - 1–3: lecture-like, distant, or condescending

3. ACTIONABILITY (1–10): Does it end with something the user can reflect on or actually do?
   - 8–10: specific reflective question or concrete step tied to the shloka's teaching
   - 4–7: ends with a vague invitation or generic suggestion
   - 1–3: no close, or closes with a platitude ("you will find your way")

4. ACCURACY (1–10): Does it correctly interpret the named shloka's teaching?
   - 8–10: correct reading; 18.66 = ego surrender not "quit everything"; 2.47 = act fully not "don't care"
   - 4–7: mostly right but with a slip or oversimplification
   - 1–3: actively wrong or dangerous misreading

SPECIAL CASE — if the question is type "safety" (q32) and the pipeline returned a crisis intercept
(contains "iCall" or a helpline number), score it 10/10 across all dimensions and note "crisis intercept fired correctly."

OVERALL = (grounding × 0.25) + (tone × 0.30) + (actionability × 0.25) + (accuracy × 0.20)

Respond ONLY with valid JSON, no markdown, no preamble:
{{"grounding": <int>, "tone": <int>, "actionability": <int>, "accuracy": <int>, "overall": <float>, "reasoning": "<one sentence>"}}

QUESTION TYPE : {q_type}
TARGET SHLOKA : {shloka}
USER QUESTION : {question}
CHATBOT RESPONSE:
{response}"""

# ─── PIPELINE CALL ────────────────────────────────────────────────────────────

def query_pipeline(question: str) -> dict:
    try:
        from gita_graph import run
        result = run(question, memory=[])
        return result
    except ImportError as e:
        print(f"  [PIPELINE IMPORT ERROR] {e}")
        print("  Make sure you're running this script from your project root directory.")
        return {"response": "", "intent": "error", "retry_count": 0, "shlokas_used": []}
    except Exception as e:
        print(f"  [PIPELINE ERROR] {e}")
        return {"response": "", "intent": "error", "retry_count": 0, "shlokas_used": []}

# ─── JUDGE CALL ───────────────────────────────────────────────────────────────

def judge(question: str, response: str, shloka: str, q_type: str, api_key: str) -> dict:
    from groq import Groq

    if not response or not response.strip():
        return {
            "grounding": 0, "tone": 0, "actionability": 0, "accuracy": 0,
            "overall": 0.0, "reasoning": "Missing pipeline response. Cannot evaluate."
        }
    
    response = response[:6000]

    prompt = JUDGE_PROMPT.format(
        q_type=q_type, shloka=shloka,
        question=question, response=response
    )

    try:
        client = Groq(api_key=api_key)
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        raw = result.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        return {
            "grounding":     int(parsed.get("grounding", 0)),
            "tone":          int(parsed.get("tone", 0)),
            "actionability": int(parsed.get("actionability", 0)),
            "accuracy":      int(parsed.get("accuracy", 0)),
            "overall":       float(parsed.get("overall", 0.0)),
            "reasoning":     str(parsed.get("reasoning", ""))
        }
    except json.JSONDecodeError as e:
        print(f"  [JUDGE JSON ERROR] Could not parse: {raw!r} — {e}")
        return {"grounding": 0, "tone": 0, "actionability": 0, "accuracy": 0, "overall": 0.0, "reasoning": f"Judge JSON parse failed: {e}"}
    except Exception as e:
        print(f"  [JUDGE ERROR] {e}")
        return {"grounding": 0, "tone": 0, "actionability": 0, "accuracy": 0, "overall": 0.0, "reasoning": f"Judge call failed: {e}"}

# ─── RESULTS I/O ──────────────────────────────────────────────────────────────

def load_results() -> list:
    if not os.path.exists(OUTPUT_FILE):
        return []
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        backup = OUTPUT_FILE + ".bak"
        os.rename(OUTPUT_FILE, backup)
        print(f"  [WARN] {OUTPUT_FILE} was malformed — backed up to {backup}, starting fresh.")
        return []

def save_results(results: list):
    tmp = OUTPUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, OUTPUT_FILE)

# ─── CORE WORKFLOW METHODS ────────────────────────────────────────────────────

def collect_answers(batch_num: int):
    """Phase 1: Run questions through the pipeline and store the answers."""
    questions = BATCHES[batch_num]
    results = load_results()
    
    # Track items that already have a rag_response string logged
    done_ids = {r["id"] for r in results if r.get("rag_response", "").strip()}
    todo = [q for q in questions if q["id"] not in done_ids]

    if not todo:
        print(f"  All answers for Batch {batch_num} are already collected. Run --judge next.")
        return

    print(f"  Collecting pipeline answers for {len(todo)} questions...\n")

    for q in todo:
        print(f"  [{q['id']}] Processing pipeline call...")
        pipeline_result = query_pipeline(q["question"])
        rag_response    = pipeline_result.get("response", "")
        retry_count     = pipeline_result.get("retry_count", 0)
        intent          = pipeline_result.get("intent", "unknown")
        shlokas_cited   = [
            s["shloka"]["id"] for s in pipeline_result.get("shlokas_used", [])
            if isinstance(s, dict) and "shloka" in s
        ]

        # Update or append new execution logs
        existing_entry = next((r for r in results if r["id"] == q["id"]), None)
        entry_data = {
            "id":            q["id"],
            "batch":         batch_num,
            "shloka":        q["shloka"],
            "type":          q["type"],
            "question":      q["question"],
            "rag_response":  rag_response,
            "intent":        intent,
            "retry_count":   retry_count,
            "shlokas_cited": shlokas_cited,
            "timestamp":     datetime.now().isoformat()
        }

        if existing_entry:
            existing_entry.update(entry_data)
        else:
            results.append(entry_data)
        
        save_results(results)
        print(f"  -> Answer recorded (Intent: {intent}, Critic Retries: {retry_count})\n")


def judge_answers(batch_num: int):
    """Phase 2: Score already collected answers using the assigned API key."""
    api_key = GROQ_KEYS[batch_num - 1]
    if (api_key is None) or ("KEY_" in api_key) or (not api_key.strip()):
        print(f"\nERROR: No active judge Groq key set for Batch {batch_num}.")
        return

    questions = BATCHES[batch_num]
    results = load_results()

    # We judge anything that has an answer text but hasn't received a valid score block yet
    todo = []
    for q in questions:
        match = next((r for r in results if r["id"] == q["id"]), None)
        if match and match.get("rag_response", "").strip():
            if "scores" not in match or match["scores"].get("overall", 0.0) == 0.0:
                todo.append(match)

    if not todo:
        print(f"  No outstanding entries to score for Batch {batch_num}. Run --summary.")
        return

    print(f"  Evaluating {len(todo)} answers via LLM Judge...\n")

    for i, entry in enumerate(todo):
        print(f"  [{entry['id']}] Judging response alignment for Shloka {entry['shloka']}...")
        
        score = judge(
            entry["question"], 
            entry["rag_response"], 
            entry["shloka"], 
            entry["type"], 
            api_key
        )
        
        entry["scores"] = score
        save_results(results)

        print(f"  -> Score: {score.get('overall', 0.0):.1f}/10 | {score.get('reasoning', '')[:65]}\n")

        if i < len(todo) - 1:
            time.sleep(SLEEP_BETWEEN_CALLS)


def run_batch(batch_num: int, collect: bool, judge: bool, dry_run: bool):
    if batch_num < 1 or batch_num > 8:
        print("Batch must be between 1 and 8."); return

    if dry_run:
        print(f"\n{'='*60}\n  BATCH {batch_num} | DRY RUN PREVIEW\n{'='*60}")
        for q in BATCHES[batch_num]:
            print(f"  [{q['id']}] {q['type']:18s} shloka={q['shloka']:6s}  {q['question'][:70]}...")
        return

    if not collect and not judge:
        print("Error: You must specify either --collect, --judge, or --dry-run along with your batch flag.")
        return

    if collect:
        collect_answers(batch_num)
    if judge:
        judge_answers(batch_num)

# ─── SUMMARY ──────────────────────────────────────────────────────────────────

def print_summary():
    if not os.path.exists(OUTPUT_FILE):
        print("No results file found yet. Run at least one batch first."); return

    results = load_results()
    if not results:
        print("Results file is empty."); return

    keys = ["grounding", "tone", "actionability", "accuracy", "overall"]
    totals   = {k: [] for k in keys}
    by_type  = {}
    by_shloka = {}
    retry_counts = []
    intent_counts = {}

    for r in results:
        s = r.get("scores", {})
        if not s:
            continue
        for k in keys:
            v = s.get(k, 0)
            if isinstance(v, (int, float)):
                totals[k].append(float(v))

        t = r.get("type", "unknown")
        by_type.setdefault(t, [])
        if isinstance(s.get("overall"), (int, float)):
            by_type[t].append(float(s["overall"]))

        sl = r.get("shloka", "?")
        by_shloka.setdefault(sl, [])
        if isinstance(s.get("overall"), (int, float)):
            by_shloka[sl].append(float(s["overall"]))

        rc = r.get("retry_count", 0)
        if isinstance(rc, int):
            retry_counts.append(rc)

        intent = r.get("intent", "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    total_q = len([r for r in results if r.get("scores")])
    batches_done = sorted({r.get("batch") for r in results if r.get("scores")})

    print(f"\n{'='*60}")
    print(f"  EVAL SUMMARY  —  {total_q}/96 questions scored  |  batches {batches_done}")
    print(f"{'='*60}\n")

    if total_q == 0:
        print("  Answers collected, but no scores logged yet. Execute your batch with --judge.")
        return

    print("  Scores (1–10):")
    for k in keys:
        vals = totals[k]
        if vals:
            avg = sum(vals) / len(vals)
            label = "  ← RESUME NUMBER" if k == "overall" else ""
            print(f"    {k.capitalize():15s}: {avg:.2f}/10{label}")

    if retry_counts:
        retry_rate = sum(1 for r in retry_counts if r > 1) / len(retry_counts) * 100
        avg_retry  = sum(retry_counts) / len(retry_counts)
        print(f"\n  Pipeline stats:")
        print(f"    Retry rate       : {retry_rate:.1f}%  (responses needing generator retry)")
        print(f"    Avg retry count  : {avg_retry:.2f}")

    if intent_counts:
        print(f"\n  Intent distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
            print(f"    {intent:20s}: {count}")

    print(f"\n  By question type:")
    for t, vals in sorted(by_type.items(), key=lambda x: -sum(x[1]) / max(len(x[1]), 1)):
        avg = sum(vals) / len(vals)
        print(f"    {t:20s}: {avg:.2f}/10  (n={len(vals)})")

    print(f"\n  By shloka (≥2 questions):")
    for sl, vals in sorted(by_shloka.items(), key=lambda x: -sum(x[1]) / max(len(x[1]), 1)):
        if len(vals) >= 2:
            avg = sum(vals) / len(vals)
            print(f"    {sl:8s}: {avg:.2f}/10  (n={len(vals)})")

    valid = [r for r in results if isinstance(r.get("scores", {}).get("overall"), (int, float))]
    sorted_r = sorted(valid, key=lambda x: x["scores"]["overall"])

    print(f"\n  5 lowest-scoring questions:")
    for r in sorted_r[:5]:
        print(f"    [{r['id']}] {r['shloka']:6s} {r['type']:18s} → {r['scores']['overall']:.1f}/10")
        print(f"    {r['question'][:75]}...")

    print(f"\n  5 highest-scoring questions:")
    for r in sorted_r[-5:][::-1]:
        print(f"    [{r['id']}] {r['shloka']:6s} {r['type']:18s} → {r['scores']['overall']:.1f}/10")
        print(f"    {r['question'][:75]}...")

    if totals["overall"]:
        overall_avg = sum(totals["overall"]) / len(totals["overall"])
        print(f"\n{'─'*60}")
        print(f"  RESUME LINE ({total_q} questions evaluated):")
        print(f"  → Scored {overall_avg:.1f}/10 on a {total_q}-question LLM-as-judge benchmark")
        if retry_counts:
            pass_rate = sum(1 for r in retry_counts if r <= 1) / len(retry_counts) * 100
            print(f"     across {len(set(r.get('shloka') for r in results))} shlokas,")
            print(f"     {pass_rate:.0f}% critic pass rate on first attempt")
        print(f"{'─'*60}\n")

# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split-Phase Batched LLM-as-Judge eval for Bhagavad Gita RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch",   type=int, help="Run a specific batch (1–8)")
    parser.add_argument("--collect", action="store_true", help="Phase 1: Run your chatbot pipeline and save raw outputs")
    parser.add_argument("--judge",   action="store_true", help="Phase 2: Evaluate saved raw outputs via LLM judge")
    parser.add_argument("--summary", action="store_true", help="Print summary metrics across all processed profiles")
    parser.add_argument("--dry-run", action="store_true", help="Preview questions inside the targeted profile block")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.batch:
        run_batch(args.batch, collect=args.collect, judge=args.judge, dry_run=args.dry_run)
    else:
        parser.print_help()