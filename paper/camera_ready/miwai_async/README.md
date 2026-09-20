# MIWAI 2026, paper 122: files for the asynchronous session

The conference is in Halifax, Nova Scotia, 10 to 12 October 2026.

The organizers wrote that every paper has to join the asynchronous session,
and that it carries "an abstract, infographic and video for each paper".

| Required | File | Status |
| --- | --- | --- |
| Abstract | `abstract.txt` | Done. 247 words, copied from the accepted paper. |
| Infographic | `infographic.png` | Draft. 1920 x 1080. |
| Video | | Not recorded yet. |

The slides and the script are not submitted on their own, but the video will
be recorded from them.

| File | What it is |
| --- | --- |
| `presentation_miwai.pdf` | The deck, 23 pages. 14 to present, 3 of references, 5 kept back for questions. |
| `TALK_SCRIPT.md` | What to say on each slide, with timings. Runs 13 to 14 minutes in a 15 minute slot. Likely questions are at the end. |
| `infographic.tex` | Source of the infographic. Build commands are in its header. |

## Where the numbers come from

All of them come from the abstract of the accepted paper: Severe recall 12.3%
to 48.6%, Severe F1 0.152 to 0.356, and eight unseen SPIDER labels at mean F1
0.362. Nothing from the thesis experiments appears here.

The 4% Severe share in the first panel is the mean of 4.8%, 4.1% and 3.9%,
which are the three sagittal conditions.

## Three things I would like your opinion on

1. The page has about 100 words. Publishers who state a limit for graphical
   abstracts say 50. That limit is written for a thumbnail in a printed
   contents page, and this one is read at full size in a browser, so I went
   over it. It is the first thing to cut if the page looks crowded.

2. I did not reuse Figure 1 from the paper. The guidance I found says a
   methods figure makes a poor infographic, because it is drawn to be read
   slowly next to the text. So the architecture here is two boxes and a
   fusion step.

3. The colours are the Okabe-Ito set, which stays readable with red-green
   colour blindness. Nothing on the page depends on colour alone.

## Still unknown

The organizers have not published a size, format or word limit for the
infographic, and have not given a deadline. I picked 1920 x 1080 so the
slides, the infographic and the video share one aspect ratio. If they publish
a specification later, re-exporting from the .tex file is quick.
