# MIWAI 2026, paper 122: materials for the asynchronous session

Everything the organizers asked for, in one place, plus the two files the
live talk needs. Conference is in Halifax, Nova Scotia, 10 to 12 October 2026.

## What the organizers require

Their email says every paper must take part in the asynchronous session, and
that the asynchronous session carries "an abstract, infographic and video for
each paper".

| Required | File | Status |
| --- | --- | --- |
| Abstract | `abstract.txt` | Ready. 247 words, taken verbatim from the accepted paper. |
| Infographic | `infographic.png` | Draft for review. 1920 x 1080. |
| Video | not here | Not recorded yet. Script and slides below are the raw material. |

## Supporting files, not submitted on their own

| File | What it is |
| --- | --- |
| `presentation_miwai.pdf` | The talk deck, 23 pages: 14 to present, 3 of references, 5 held in reserve for questions. |
| `TALK_SCRIPT.md` | Word for word speaking script, timed per slide, lands at 13 to 14 minutes inside a 15 minute slot. Also carries the anticipated questions and their answers. |
| `infographic.tex` | Source of the infographic. Rebuild with the two commands in its header comment. |

## Notes for the reviewer

**Every number on the infographic comes from the abstract of the accepted
paper.** None of it reports the later thesis experiments, which are a separate
body of work and are not part of this paper.

- Severe Recall 12.3% to 48.6%, and Severe F1 0.152 to 0.356, on RSNA 2024.
- Eight unseen SPIDER labels graded zero-shot, mean F1 0.362.
- The Severe share of 4% in panel 1 is the mean over the three sagittal
  conditions, whose individual values are 4.8%, 4.1% and 3.9%.

**Design choices worth a second opinion.**

- The colours are the Okabe-Ito set, which stays legible under the common
  forms of colour vision deficiency. No meaning is carried by colour alone.
- The architecture is deliberately reduced to two boxes and a fusion step.
  Published guidance on graphical abstracts is consistent that reusing a
  paper's real architecture figure makes a poor infographic, because a methods
  figure is built to be read slowly alongside the text.
- The page carries roughly 100 words of body text. Publisher guidance for
  journal graphical abstracts suggests staying under about 50. That guidance
  targets a small thumbnail in a printed table of contents; this one is meant
  to be read at full size in a browser. It is still the most likely thing to
  cut if it reads as crowded.

**Open question for the conference organizers, not settled here:** they have
published no size, format or word limit for the infographic that we could
find. 1920 x 1080 was chosen so the same aspect ratio serves the slides, the
infographic and the video. If they publish a specification, the source file
makes it cheap to re-export.
