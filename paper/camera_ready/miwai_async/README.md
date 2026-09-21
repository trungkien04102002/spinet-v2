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

Every figure is traced to the accepted paper.

| On the page | In the paper |
| --- | --- |
| Severe recall 12.3% to 48.6% | Abstract, and Table 1, mean over seeds 42, 123, 456 |
| Severe F1 0.152 to 0.356 | Abstract, same table |
| about four times higher | Section 1, "about 4x" |
| disc morphology F1 0.587 against 0.433 | Table 4, row "Mean (disc-morphology)" |
| 81% / 15% / 4% | Mean of the per-condition values behind Figure 2, which are 4.8, 4.1 and 3.9 for Severe |

Nothing from the thesis experiments appears here.

**One choice you should check.** My first draft headlined the zero-shot result
as mean F1 0.362 over all eight labels. That is the wrong number to show. The
paper's own Table 4 gives off-the-shelf BiomedCLIP 0.394 on the same all-eight
mean, so it beats us there, and Section 4.4 says so in plain words. Putting
0.362 on the page as an achievement would have contradicted our own paper.

The page now shows the disc morphology mean instead, 0.587 against 0.433,
which is the comparison the paper does claim, and it names the three labels it
covers. If you would rather the infographic carry the all-eight number with
the loss stated honestly, or carry no zero-shot number at all, say so and I
will change it.

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
