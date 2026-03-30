# Power Dynamics Annotation Guidelines

## West Wing Dialogue Corpus

**Prepared by:** Nathan Subrahmanian, Galileo Steinberg, Shai Goldberg-Kellman

---

## 1. Project Background and Goals

### About the Show

*The West Wing* is a political drama television series (1999–2006, 7 seasons) set in the White House. The show follows President Josiah "Jed" Bartlet and his senior staff as they navigate political crises, policy debates, and personal relationships. The cast includes the President, his Chief of Staff Leo McGarry, Communications Director Toby Ziegler, Deputy Communications Director Sam Seaborn, Deputy Chief of Staff Josh Lyman, Press Secretary C.J. Cregg, and their various assistants, family members, and political contacts.

### Why This Dataset?

The West Wing is an ideal corpus for studying conversational power dynamics for several reasons. The show features characters with well-defined hierarchical relationships (President → Chief of Staff → senior staff → assistants) that are frequently subverted in actual dialogue. A character with formal authority does not always dominate a conversation. Subordinates regularly push back, manage up, redirect, or control the flow of discussion. This creates a rich dataset where power in conversation cannot be predicted from titles alone.

### Project Goal

We aim to annotate the power dynamics present in paired character dialogues from the show. Using these annotations, we can study which characters tend to be more assertive in their interactions, how power dynamics shift across the course of the series, and what conversational strategies characters use to assert or yield control. Downstream tasks may include building models to predict power dynamics from dialogue features and statistically testing how character relationships evolve over time.

---

## 2. Task Overview

You will be annotating short excerpts of dialogue between exactly two characters from the show. Each excerpt contains between 6 and 20 conversational turns (a "turn" is one character speaking).

For each excerpt, you will complete **three annotation tasks**, in order of importance:

1. **Power Dynamic Rating (required):** A 5-point scale indicating which character is more assertive or dominant in the conversation. This is the primary annotation and must be completed for every valid document.

2. **Power Shift Flag (secondary):** A yes/no flag indicating whether the dominant character changes during the excerpt.

3. **Power Expression Checklist (secondary):** A checklist of conversational strategies used to express power. Check all that apply.

The rating scale is the essential annotation. The flag and checklist provide additional granularity and context, but the scale is the foundation of the dataset.

---

## 3. Before You Annotate: Document Validity Check

Our dialogue excerpts were generated automatically by an algorithm that splits full episode scripts into paired character exchanges. Occasionally, the algorithm produces invalid or problematic documents. **Before annotating each document, check the following:**

**A valid document must:**

- Contain dialogue from **exactly two characters** (the two listed in the metadata header)
- Contain **actual back-and-forth dialogue** (not just one character speaking with no response)
- Be **coherent and readable** — the text should make sense as a conversation, not be garbled or cut off mid-word in a way that makes the exchange impossible to interpret

**If you encounter an invalid document**, do not annotate it. Instead, **skip it and flag it** by recording the document's metadata (episode, characters, pair instance, excerpt number) in your annotation notes so we can review it later.

Minor imperfections are expected and should **not** cause you to skip a document. These include: stage directions appearing between lines, slight formatting inconsistencies, a character's name appearing with slight variations (e.g., "SAM SEABORN" in one line and "SAM" in later lines), or the final line being slightly cut off but the overall exchange still being clear.

---

## 4. Primary Annotation: Power Dynamic Rating

For each valid document, you will assign a rating on the following scale. **Character A** and **Character B** refer to the characters listed in the document's metadata header as `character_a` and `character_b`.

| Rating | Label | Meaning |
|--------|-------|---------|
| **-2** | **A Clearly Dominates** | Character A is clearly the more assertive, controlling, or dominant figure in this exchange. Character B mostly defers, yields, or follows A's lead. |
| **-1** | **A Somewhat More Assertive** | Character A has a slight edge in assertiveness. They may set more of the agenda or push back more firmly, but B is not entirely passive. |
| **0** | **Balanced** | Neither character is clearly more assertive. This includes conversations where both characters are collaborating as equals, casual/friendly exchanges where no one is trying to dominate, OR conversations where both characters are pushing back on each other with roughly equal force. |
| **+1** | **B Somewhat More Assertive** | Character B has a slight edge in assertiveness. They may set more of the agenda or push back more firmly, but A is not entirely passive. |
| **+2** | **B Clearly Dominates** | Character B is clearly the more assertive, controlling, or dominant figure in this exchange. Character A mostly defers, yields, or follows B's lead. |

---

### CRITICAL NOTE: Annotate the Text, Not the Characters

**You must base your rating entirely on what happens in the dialogue you are reading.** Do not let your knowledge of the show, the characters' job titles, or their general relationships influence your annotation. The President of the United States can be the less assertive character in a given conversation. An assistant can dominate an exchange with their boss.

If you know the show, set that aside. If you don't know the show, that's fine. You have everything you need in the text. Judge **only** by what the characters say and do in the excerpt in front of you.

---

### Rating Explanations and Examples

#### -2 / +2: One Character Clearly Dominates

Use this when one character is unmistakably in control of the exchange. Signs include:

- One character gives orders and the other follows them
- One character repeatedly shuts down or dismisses the other
- One character interrogates and the other answers defensively
- One character's words consist mostly of brief compliance ("Yes sir," "Okay," "You're right")

**Example (would be rated -2, A clearly dominates):**

> *Character A: LEO, Character B: PHIL — S01E22*
>
> PHIL: Mr. President, can I suggest that rather than jumping into a military rescue mission —
>
> LEO: Oh, please.
>
> PHIL: That a phone call to the Iraqi ambassador —
>
> LEO: Bill, you wanna check with the embassy?
>
> PHIL: I'm saying that three hours spent on diplomatic solutions —
>
> LEO: I'll tell you what, Phil. How about I drop you and your 47 million-dollar warplane that's already been picked up by Iraqi radar in the middle of the desert. Then you tell me if we've got three hours to find a diplomatic solution before we come and get you!

Leo dismisses Phil's suggestions multiple times, interrupts him, and finishes with a forceful rhetorical put-down. Phil never successfully completes a point.

**Example (would be rated +2, B clearly dominates):**

> *Character A: DONNA, Character B: JOSH — S01E02*
> 
> DONNA: You're with the Energy Secretary in five minutes.
>
> JOSH: Thanks.
>
> DONNA: What's going on?
>
> JOSH: Nothing.
>
> DONNA: Really?
>
> JOSH: Yes.
>
> DONNA: You're lying?
>
>JOSH: Yes.
>
> DONNA: So I should get out?
>
> JOSH: Yes. [Donna leaves] Look, whatever quest...

In this case, Josh clearly dominates the conversation. He holds all of the information and refuses to tell Donna what is going on, placing her in a submissive position. She also asks whether she should leave, which shows her lack of control in this situation. Lastly, Josh's responses are short and curt, indicating that he is not interested in having an extended conversation with Donna.

#### -1 / +1: One Character Somewhat More Assertive

Use this when one character has a noticeable edge but the exchange isn't one-sided. The less assertive character still pushes back, contributes meaningfully, or holds their own in parts of the conversation. But on balance, one character sets the tone, directs the conversation, or exerts more influence.

**Example (would be rated +1, B somewhat more assertive):**

> *Character A: JOSH, Character B: LEO — S01E02*
>
> LEO: We need her again, Josh.
>
> JOSH: Mandy?
>
> LEO: We need her.
>
> JOSH: Wait a second. This is an ambush.
>
> LEO: Can you think of a single reason not to use Mandy that isn't personal?
>
> JOSH: Sure.
>
> LEO: What?
>
> JOSH: She used to be my girlfriend!
>
> LEO: That's good enough for me. Let's do it.

Leo is driving the agenda and ultimately makes the decision, but Josh pushes back with humor and genuine objections. It's not a total steamroll — Josh gets his say — but Leo clearly controls where this ends up.

**Example (would be rated -1, A somewhat more assertive):**

> *Character A: C.J., Character B: DANNY — S01E03*
>
> DANNY: I obviously don't have enough for a story, but as a courtesy to you C.J. I just wanted to let you know I'm gonna be asking around.
>
> C.J.: Danny, it's gonna be much ado about nothing.
>
> DANNY: It doesn't look that way.
>
> C.J.: But it is that way and I just got through telling you it's that way.
>
> DANNY: C.J.
>
> C.J.: Sam knows the difference between right and wrong and so do you...

C.J. is more assertive. She's deflecting Danny's inquiry, making declarative statements, and pushing back firmly. But Danny isn't passive either; he's pressing his case and not backing down. C.J. has the edge, not total control.

**Example (would be rated -1, A somewhat more assertive):**

> *Character A: DONNA, Character B: JOSH — S01E01*
>
> JOSH: No.
>
> DONNA: Put it on.
>
> JOSH: No.
>
> DONNA: Put it on.
>
> JOSH: No.
>
> DONNA: You've been wearing the same clothes for 31 hours now, Josh.
>
> JOSH: I am not getting spruced up for these people, Donna.
>
> DONNA: All the girls think you look really hot in this shirt.
>
> *Josh grabs the shirt and tie.*

Despite Josh's repeated refusals, Donna persists and ultimately gets him to comply. She controls the outcome of the interaction. This would be a tricky case since you might think that A clearly dominates, but it is closer to balanced since there is a butting of heads. Both characters are vying for control.

#### 0: Balanced

Use this for conversations where neither character clearly has the upper hand. This covers **two very different situations**. The checklist (Section 6) is where you'll note whether the balance feels collaborative or contentious.

**Balanced-collaborative:** Both characters are on the same page, neither directing the other. Casual conversation, joint problem-solving, or friendly banter.

**Example (balanced-collaborative):**

> *Character A: BARTLET, Character B: JOSH — S01E06*
>
> BARTLET: Truth be known, I don't have any cash on me.
>
> JOSH: It's fine.
>
> BARTLET: I don't carry cash anymore. I don't carry keys either.
>
> JOSH: Well, I wouldn't think you'd need them sir.
>
> BARTLET: How's it going in there?
>
> JOSH: We'll see.
>
> BARTLET: I appreciate it.
>
> JOSH: Yes sir.

This is a casual, warm exchange. Bartlet and Josh are chatting as friendly colleagues. No one is directing or deferring, they're just talking. You might think to annotate this as -1 since Josh consistently calls Bartlett "sir," but there is no domination in this excerpt. The title is a mark of respect and formality more than deference.

**Balanced-contentious:** Both characters are asserting themselves with roughly equal force. Neither one yields or controls the direction.

**Example (balanced-contentious):**

> *Character A: JOSH, Character B: MANDY — S01E01*
>
> JOSH: You're dating Lloyd Russell.
>
> MANDY: Yes.
>
> JOSH: Wow. That's great... I always thought he was gay.
>
> MANDY: No you didn't.
>
> JOSH: I did.
>
> MANDY: He's not gay.
>
> JOSH: You sure?
>
> MANDY: Very sure... Josh, take me seriously.
>
> JOSH: I do.
>
> MANDY: The New York Times is gonna release a poll... that brings your unfavorables up to 48%.
>
> JOSH: This is the first I'm hearing of it.
>
> MANDY: I'm here for a while. And I want you at your fighting weight when I start bitch-slapping you guys around the beltway.

Both characters are sparring. Josh needles Mandy; Mandy fires back with a political threat. Neither one backs down. This is a **0** because it is balanced, but combative. This document is a little tricky because Josh takes some control when Mandy asks him to take her seriously, but the dynamic shifts back in the last line. Overall, this is an example of a **balanced-contentious** exchange.

---

### Edge Cases and How to Decide

**When humor masks power:** Characters in this show frequently use sarcasm, jokes, and wit. Humor can be a tool of dominance (using a joke to shut someone down or deflect) or a sign of comfort and equality (mutual banter). Ask yourself: *does the humor give one character control over the conversation, or are they both enjoying it?*

> *Character A: BARTLET, Character B: TOBY — S01E05*
>
> BARTLET: I'm playing.
>
> TOBY: Mr. President, there's no shame in calling it quits. All you have to do is say, 'Toby, you're the superior athlete,' and slink on off the court.
>
> BARTLET: Take the ball out, Toby.
>
> TOBY: Let the poets write that he had the tools of greatness, but the voices of his better angels was shouted down by his obsessive need to win.
>
> BARTLET: You want to play or write my eulogy?

Both characters are using humor aggressively. Toby needles the President; the President fires back. Neither backs down. This could be argued to be a **0** (balanced-contentious) since it is not a case of either character dominating through humor, or a **1** because Bartlet seems to be more on the defensive with his shorter replies.

**When care equals control:** Sometimes a character exercises power by managing, caretaking, or mothering another character. If one character tells another what to eat, what to wear, or how to behave, and the other complies, then that's assertiveness, even if it comes from a place of caring.

> *Character A: CHARLIE, Character B: MRS. LANDINGHAM — S01E19*
>
> CHARLIE: The President'd prefer a sandwich. He says roast beef would be fine.
>
> MRS. LANDINGHAM: It's a salad, Charlie.

Mrs. Landingham is overruling the President's preferences (relayed through Charlie). The caretaking is a form of dominance in the conversation.

**Short or ambiguous excerpts:** Some documents are very brief. If the exchange is too short to determine a clear dynamic, lean toward **0**. Don't overthink short documents, go with your gut.

**Stage directions:** Stage directions (e.g., "Josh walks away," "She looks down") are part of the document and can inform your annotation. If a stage direction shows a character physically complying (grabbing the shirt, leaving the room when told) or asserting (stopping someone, slamming a door), factor that in.

**One character talks much more:** Speaking more doesn't automatically mean dominance. A character who gives a long explanation may be in a position of authority (lecturing, instructing) or in a position of weakness (justifying, pleading). Consider the *function* of the extended speech, not just its length.

---

## 5. Secondary Annotation: Power Shift Flag

After assigning your rating, answer this yes/no question:

> **Does the dominant character change during this excerpt?**

Mark **Yes** if the conversation begins with one character being more assertive but ends with the other taking control. Mark **No** if the dynamic is consistent throughout, or if the exchange is balanced throughout.

**Example of a shift:**

> *Character A: LAURIE, Character B: SAM — S01E02*
> 
> SAM: Who's Brittany?
>
> LAURIE: I am.
>
> SAM: Okay.
>
> LAURIE: Sam, we're in the middle of something here...
>
> SAM: No problem. I don't mean to interrupt. I'll just go back to the bar and call
my friend, the Assistant U.S. Attorney General, and see if he wants to come
down and meet for a drink with me and that woman back there.
>
> LAURIE: Excuse me. [leaves abruptly]
>
> SAM: [to man] Good to meet you. [to the other man] Okay.
>
> CUT TO:EXT. WASHINGTON D.C. STREET - NIGHT
> Laurie comes out of the restaurant. Sam is walking behind her.
>
> SAM: Laurie. I called you four times. You said you were gonna call me back.
>
> LAURIE: Stay away from me.
>
> SAM: Laurie?
>
> LAURIE: I can't believe you just did that!
>
> SAM: I came here in the spirit of...
>
> LAURIE: I left my jacket at the table. I can't go back there.
>
> SAM: If I cost you some money, I'll write you a check.
>
> LAURIE: You go to hell for saying that.
>
> SAM: I wasn't... [Laurie walks off ahead of Sam. He follows behind.] I'm sorry.
That was the wrong thing to say.
>
> LAURIE: Yes.
>
> SAM: You're gonna freeze out here.
>
> LAURIE: I don't care.

In this document, Sam starts out much more assertive than Laurie by taking control of the situation and even threatening Laurie. She defers to his wants and leaves the restaurant, so this would ordinarily be a clear case of **2** since Sam dominates clearly. However, once they get outside, Laurie becomes the assertive character and Sam is left in a position of apologizing and pleading. For this example, you would mark that the dynamic shifted.

You do not need to identify exactly where the shift occurs, just note that one happened.

---

## 6. Secondary Annotation: Power Expression Checklist

For each document, check **all strategies that apply** regardless of which character uses them. You do not need to specify which character uses which strategy — just note that the strategy is present in the conversation.

| Strategy | Description |
|----------|-------------|
| **Direct orders or instructions** | One character tells the other what to do. ("Walk with me." / "Write me the answer on Cuba." / "Put it on.") |
| **Controls information** | One character decides what gets discussed, reveals information strategically, or withholds information. ("I'm not talking about this." / "We need her again, Josh.") |
| **Dismisses or shuts down** | One character rejects, ignores, or talks over the other's point. ("Oh, please." / One-word rejections like "No.") |
| **Interrogates or corners** | One character uses questions to pressure, challenge, or trap the other. ("Can you think of a single reason that isn't personal?" / "Did she know who you are?") |
| **Appeals to authority or rank** | A character invokes their position, invokes someone else's authority, or uses formal address to reinforce hierarchy. ("sir" / "the President ordered me to..." / invoking a boss's wishes) |
| **Humor or sarcasm to assert** | A character uses wit, jokes, or sarcasm to deflect, shut down, or gain the upper hand. ("1-800-BITE-ME" / "I always thought he was gay.") |
| **Manages or caretakes** | A character directs another's personal behavior — what to wear, eat, or do — and the other complies. This is assertiveness through care. ("You've been wearing the same clothes for 31 hours." / "Take your back medicine.") |
| **Emotional pressure or reprimand** | A character uses guilt, disappointment, or moral authority to influence the other. ("I was really offended, too." / "I want to make sure you're taking me seriously.") |

If none of these clearly apply — for example, in a very brief, neutral exchange — it is fine to leave the checklist empty.

---

## 7. Annotation Workflow Summary

For each document:

1. **Check validity.** Does it have exactly two speakers, actual back-and-forth, and readable text? If not, skip and flag it.
2. **Read the full excerpt.** Read all the way through before deciding.
3. **Assign the power dynamic rating** (-2 to +2).
4. **Mark the power shift flag** (yes/no).
5. **Check applicable power expression strategies** from the checklist.

---

## 8. Additional Notes

**Pace yourself.** Most excerpts should take 1–2 minutes to annotate. If you find yourself deliberating for much longer, go with your instinct. The scale is designed so that reasonable people will sometimes disagree between adjacent values (e.g., -1 vs. -2). That's expected and fine.

**Don't second-guess across documents.** Each document is independent. If you see the same character pair across multiple excerpts, the dynamic may genuinely be different in each one. Annotate what you see in each individual excerpt.

**Stage directions are context, not dialogue.** Use them to inform your understanding (a character slamming a door tells you something about the emotional tone), but weight your annotation primarily toward what the characters say.

**When in doubt between adjacent ratings,** ask yourself: *if I described this conversation to a friend, would I say one character was in charge, or would I say it was pretty even?* If your friend would say "yeah, one of them was running that conversation," you're probably looking at a -2/+2 or -1/+1. If your friend would say "they were both going at it" or "just a normal chat," you're probably at 0.

**This is a fan transcription, not an official script.** Expect minor imperfections: occasional typos, inconsistent formatting of stage directions, or slight transcription errors. These should not affect your ability to annotate.