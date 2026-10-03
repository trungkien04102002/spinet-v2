# Speaking script, MIWAI 2026 paper 122

Online, live, 15 minutes, on **Saturday 10 October, 8 pm Vietnam time** (10 am Halifax, UTC-3).
Deck is 25 pages: 15 to talk over, 4 of references, 1 thank-you, 5 backup. References now come **before** the thank-you slide.

The spoken lines come to about 1,840 words. At a normal delivery with the pauses marked by `/`
that is **about 14 minutes**; at your faster pace it is closer to **12 minutes**. The per-slide
budgets below are computed from those word counts, not guessed.

Rewritten on 3 October to read like a talk rather than a list. Every slide now ends with a line
that hands over to the next one, and the voice is first person throughout: "let me show you",
"I want to be honest about this", "what I find most interesting". No claim and no number changed
in that pass; the body text was tightened to pay for the transitions.

Slides 6, 8 and 11 keep a deliberately slow budget: the architecture and the prompt mechanism are
the two slides the talk exists to deliver, and the results table should not be read cell by cell.

That leaves two to three minutes of questions inside a 15-minute slot, and the five backup
slides exist for exactly that. Do not rush to fit. Take the pauses.

Rules for delivery: short sentences, slow, pause at every `/`. Say numbers slowly. If you lose your place, look at the slide title and continue. Nobody minds a pause.

---

## Slide 1. Title (20 seconds)

> Hello everyone, / and thank you for having me. / My name is Trung Kien Ha, / from Ho Chi Minh City University of Technology, in Vietnam, / and this is joint work with my advisor, / Doctor Trong Nhan Phan.
>
> Today I would like to show you a two-branch model / for lumbar disc grading, / and how we use it to grade a label set / it was never trained on.
>
> Here is where we are going.

---

## Slide 2. Outline (12 seconds)

> I will go through five parts: / the problem, / the work this builds on, / our method, / the experiments, / and what we conclude.
>
> First, the problem.

*(Do not read the list. Say this and move on.)*

---

## Slide 3. Two open problems (40 seconds)

> We work on lumbar spine MRI. / Today a radiologist grades every disc by hand. / That is slow, / and if you ask two readers / you often get two different grades.
>
> On the right you can see the data. / Normal or Mild on the left, / Severe on the right. / Canal on top, / foramen below.
>
> There are two open problems here. / The first is class imbalance. / Severe is the class the doctor cares about most, / but it is under five percent of the data, / and a model trained normally / almost never predicts it.
>
> The second is the label space. / You train on one label set, / and if someone brings you a new one, / you need a new head / and you train again.
>
> That is what we are up against. / Here is what we did.

---

## Slide 4. Contributions (30 seconds)

> Four contributions.
>
> The first is a finding: / attention helps the Severe class in-domain, / but it hurts when you move to another dataset. / A frozen prior brings it back.
>
> The second is the design: / two branches that turn out to be complementary, / not redundant.
>
> The third is zero-shot label extension. / In our model the label set becomes an input.
>
> And the fourth is a training recipe for imbalance / that lifts Severe recall / from twelve point three percent to forty-eight point six.
>
> Before I get to the method, / it is worth seeing / what this stands on.

---

## Slide 5. Related work (45 seconds)

> Three lines of work lead here.
>
> First, disc grading with convolutional networks: / SpineNet, then SpineNetV2. / We reuse their three-D ResNet-34 backbone, / weights unchanged. / Their limit is that reading only sagittal images / puts a ceiling on the foraminal conditions.
>
> Second, attention for imbalance: / CBAM, applied to lumbar stenosis. / We take it into full three-D. / Its limit is the one I mentioned: / it helps in-domain, / and hurts on transfer.
>
> Third, biomedical vision-language models, / BiomedCLIP and RadCLIP. / We reuse the frozen embedding / and read it with cosine similarity. / Their limit is that they are not volumetric, / and they are fine-tuned.
>
> Now, all three train a fixed head on one schema. / None can take a new label set as an input. / That is the gap we are trying to fill.
>
> With that gap in mind, / here is the model.

*(Do not read the table cell by cell. Say the three lines, then the gap sentence. The gap sentence is the one that matters.)*

---

## Slide 6. Architecture (1 minute 45), the core slide, go slow

> One disc volume goes in on the left, / and two branches read it.
>
> The top branch is the attention branch, / a three-D ResNet-34 with CBAM. / It looks at the shape of the disc, / and the attention helps it focus / on the small region where the lesion is.
>
> The bottom branch is the multimodal branch. / Here we use BiomedCLIP, / and we keep it frozen. / It was pretrained on millions / of medical image and text pairs, / so it already knows a great deal / about what medical images look like.
>
> We take the two vectors, / and a small MLP mixes them into one.
>
> Now here is the part I would like you to look at. / On the right / we write every label as a text prompt. / The text encoder turns each one into a vector. / We compare our image vector / against each label vector / by cosine similarity, / and the closest one is the prediction.
>
> Both backbones stay frozen, / so we only train that small head in the middle.
>
> That is the architecture. / Which brings me back / to the first problem: / imbalance.

---

## Slide 7. Class imbalance (35 seconds)

> For imbalance we use four things together.
>
> First, focal loss, / which puts more weight on the hard examples.
>
> Second, square-root class weights. / Without this the gradient ratio / between Severe and Normal / is about fifteen to one; / with it, about four. / Training is much more stable.
>
> Third, oversampling, / so the model sees Severe cases more often. / And fourth, augmentation: / flips, small rotations, / intensity changes and noise.
>
> Why does the combination work? / The frozen branch gives a stable prior / for a class we barely have examples of, / and the attention branch finds the lesion.
>
> Now the part I find most interesting: / the label space.

---

## Slide 8. Zero-shot label extension (1 minute 25)

> This is really the main idea of the paper.
>
> Phi is the encoder. / It turns one disc volume into one vector. / That part never changes. / Only the head changes.
>
> Now look at t-k. / We write a prompt for each class. / On RSNA the classes are the grades, / so for the spinal canal / we write three sentences, one per grade: / "severe spinal canal stenosis", and so on.
>
> Psi is the frozen text encoder. / It turns each sentence into a vector, / divided by its length, / so every class sits on the unit sphere.
>
> We do this once, before training. / At test time / the volume is the only thing that goes in. / The prompts are not fed in beside it. / They are the class definitions.
>
> Then we take the dot product / between the image vector and each label vector. / Both are unit length, / so that is a cosine similarity, / and the highest score wins.
>
> Our classifier, then, / is not a matrix of weights. / It is a list of sentences. / For SPIDER we simply write eight different ones.
>
> A new label is just a new prompt. / No new head, / no retraining.
>
> That is the method. / Now, how we tested it.

---

## Slide 9. Experimental setup, data (35 seconds)

> Two datasets.
>
> RSNA 2024 is what we train on, / and also our in-domain test. / Sagittal T2 and STIR. / RSNA has five conditions; / we use the three readable on sagittal images, / the spinal canal / and left and right foramen. / The other two really need axial. / Each is graded into three levels, / and we validate on about nineteen hundred discs.
>
> SPIDER we hold out entirely. / Different hospital, / different label set, / eight labels. / We use it twice: / once zero-shot, / once with supervised transfer.
>
> And as you can see on the right, / SPIDER is skewed as well.
>
> A word on how we trained.

---

## Slide 10. Experimental setup, training (30 seconds)

> This table has the details; / let me pick out three things.
>
> The input is nine slices, / one hundred and twelve by two hundred and twenty-four. / Both backbones are frozen, / and we train one point two million parameters / out of two hundred and sixty million. / AdamW, / batch thirty-two, / at most twenty epochs with early stopping.
>
> We compare four models: / the SpineNetV2 backbone, / CBAM only, / multimodal only, / and our full hybrid. / Everything you will see / is the mean over three seeds.
>
> Now the results.

---

## Slide 11. In-domain results (1 minute 30), do not read every cell

> This is our main result table. / I will not read every cell, / so let me point you at two rows.
>
> First, Mean F1. / SpineNetV2 gets zero point four two, / our hybrid gets zero point five three, / and notice that both single branches / sit in between. / The two branches together, then, / really are better than either alone.
>
> Now the bottom half, / the Severe class, / which is what matters clinically. / Severe recall goes from zero point one two / to zero point four nine, / about four times better, / and Severe F1 from zero point one five / to zero point three six.
>
> One number goes the other way, / and I want to be honest about it. / Mean accuracy drops about nine points. / The baseline gets its accuracy / by predicting Normal almost every time, / and we are trading some of that / to catch more Severe cases. / In the clinic, / missing a Severe case / is worse than a false alarm.
>
> Those F1 numbers may still look low to you. / They deserve some context.

---

## Slide 12. Putting the Severe metrics in context (1 minute)

> The fair question being: / is the model just guessing? / We do not think so, / and there are two reasons.
>
> AUPRC has a floor, / and that floor is the share of positives. / Severe is about five percent, / so random ranking scores about zero point zero five. / We get zero point three three, / roughly six point seven times chance.
>
> AUC does not have that problem; / it does not move with prevalence. / Ours is zero point nine, / where a coin flip is zero point five.
>
> The model ranks Severe cases well. / What is hard is the threshold, / not the ranking.
>
> On the right / is a Grad-CAM map for a Severe canal case. / Without attention / the response sits on the vertebral body, / away from the canal. / With CBAM / it moves onto the narrowed canal.
>
> I should say this is qualitative support only. / A saliency map tells you where the evidence sits. / It is not by itself / proof of a mechanism.
>
> That is in-domain. / The real test / is a dataset the model has never seen.

---

## Slide 13. Zero-shot transfer to SPIDER (1 minute 5)

> Same model, / new prompts for eight SPIDER labels, / nothing retrained.
>
> For the disc-shape labels / we get zero point five nine, / against zero point four three / for BiomedCLIP on its own. / So the RSNA training does transfer.
>
> But let me be honest. / Across all eight labels / we get zero point three six / and it gets zero point three nine. / It works / when the new label is close to something we trained on, / and not when it is far.
>
> And the weak labels are weak / for reasons we can name. / Spondylolisthesis is about vertebral alignment, / not disc shape. / The endplate labels are about bone. / Pfirrmann is a five-class grade / and we trained on three.
>
> The weakest is Modic, / and that one is not a model failure. / Modic grading needs a paired T1 sequence, / and SPIDER only gives us T2. / The information is simply not there.
>
> In the chart, / blue is ours, / grey is BiomedCLIP off the shelf.
>
> That was without training. / What if we do allow it?

---

## Slide 14. Supervised transfer on SPIDER (30 seconds)

*(Do not cut this slide. It is the only place in the talk where the CBAM-drops-below-baseline evidence is actually shown, and contribution 1 on slide 4 rests on it. If you must save time, take it from the outline or the contributions slide.)*

> Here we do train on SPIDER, / so this is transfer learning, / not zero-shot.
>
> Mean F1 again. / Our hybrid gets zero point six five three, / SpineNetV2 zero point six four six. / That is parity, / and I will not claim more.
>
> But CBAM on its own drops to zero point six one nine, / below the plain baseline. / Attention by itself overfits RSNA, / and the frozen branch brings it back. / That is why we call it a regularizer.
>
> With that, / let me close.

---

## Slide 15. Take-aways, future work, clinical use (55 seconds)

> Three take-aways.
>
> The imbalance recipe recovers the Severe class, / with recall about four times higher. / Labels can be an input: / a new label set is just new prompts. / And the frozen branch behaves like a regularizer, / bringing attention back / when it drops on a new dataset.
>
> One honest note. / We ran three seeds, / so our p-values are exploratory, / not a powered test.
>
> Three things we would like to do next. / An explicit image-text alignment loss. / A native three-D medical vision-language backbone, / since ours is frozen and two-D. / And axial T2 alongside sagittal, / for the two foraminal conditions / where we think the input is the real limit.
>
> And how we see it used. / Severe recall is forty-eight percent, / where screening would need about eighty. / We see this as a second reader / that flags candidate Severe cases / for the radiologist, / not a standalone tool.
>
> That is everything. / Thank you very much for listening, / and I am happy to take questions.

---

# Questions: prepared answers

**Q. Three seeds is not enough for significance.**

> You are right. / We report it as exploratory. / With three seeds / we cannot claim a powered test. / More seeds is future work.

**Q. Why should cosine similarity work for a new prompt?**

> During training / we already score by cosine / against the RSNA text prompts. / So the image vector is pulled / toward the text space. / For a new label / it works if the label is close / to something we trained on. / The backup per-label slide shows both sides. / We win on disc-shape labels / and we lose on distant labels. / We do not claim it works for everything.

**Q. What is the exact training loss?**

*(Go to page 24, "Backup: full training objective".)*

> Focal loss on the cosine logits. / Plus a supervised contrastive term, / weight zero point one. / And the tasks are combined / by learned uncertainty weighting.

**Q. Is it fast enough for clinical use?**

> Training is cheap. / We only train one point two million parameters. / But inference is not light. / Both frozen backbones still run. / About sixty-six milliseconds per disc.

**Q. Does the Grad-CAM figure prove the model looks at the right place?**

> It is qualitative support, / not a proof. / The map shows where the evidence sits, / and with CBAM it sits on the narrowed canal, / which agrees with the Severe gains in the table. / But a saliency map alone / is not proof of a mechanism, / and we say so in the paper.

**Q. Why BiomedCLIP and not another model?**

> It is public, / and it gives a shared image and text space. / That is what we need for prompts. / Comparing other medical vision-language models / is future work.

**Q. Your all-eight zero-shot mean is 0.362, but off-the-shelf BiomedCLIP gets 0.394 with no training at all. Doesn't your training hurt transfer on average?**

> On the all-eight average, yes, the off-the-shelf model is ahead. / We report that in the paper. / But that average is not the claim.
>
> The claim is that the label set can be an input at all. / A fixed-head model scores nothing on these eight labels, / because it has no head for them.
>
> And where the new labels are close to what we trained on, / the disc morphology group, / we are clearly ahead: / zero point five nine against zero point four three. / Where they are far away, / spondylolisthesis for example, / we are worse. / RSNA supervision helps transfer that is semantically near, / and does not help transfer that is far.

**Q. Why are the first few labels high and the rest low?**

> Because they are not the same kind of label.
>
> The three we do well on / are disc morphology: / bulging, herniation, narrowing. / That is what RSNA taught us. / Canal stenosis and foraminal narrowing / are both about a disc / narrowing a space. / So those prompts land / near what the model already knows.
>
> The weak ones are different problems. / Spondylolisthesis is vertebral alignment. / The endplate labels are bone. / Pfirrmann asks for five grades / where we trained on three. / And Modic needs a T1 sequence / that this dataset does not contain.
>
> I would say that is the useful part. / The model fails where you would predict / it should fail. / If it scored the same on every label, / I would not trust the result.

**Q. Do the in-domain gains survive a significance test?**

> We did not run one on that table. / Our only p-values are on the SPIDER transfer results. / With three seeds / I would not claim statistical significance in-domain. / What I can say / is that the Severe Recall gap / is much larger than the seed spread.

**Q. Does this work for the foraminal conditions too, or only the canal?**

> Not equally. / Severe recall for the two foraminal conditions / sits between eighteen and twenty-two percent. / That is well below the canal.
>
> External validation of SpineNetV2 / reports the same ceiling / for sagittal-only foraminal reading. / So we read it as a limit of the input, / not of the attention block. / Those conditions need richer input. / That is future work.

**Q. Is a frozen vision-language model worth it, against something cheap like threshold tuning?**

> That is a fair challenge, / and we do not have that exact ablation. / We do ablate the two branches separately, / and each one alone is worse than the pair. / A threshold study on the plain baseline / would be a good addition.

**Q. Can this be used in a hospital today?**

> Not alone. / Severe recall is forty-eight percent. / That is too low for screening by itself. / We see it as a second reader, / to help the radiologist.

---

# Words to practise

- BiomedCLIP: "bio-med-clip"
- cosine similarity: "co-sine si-mi-la-ri-ty"
- foraminal: "fo-ra-mi-nal"
- Pfirrmann: "feer-man"
- radiologist: "ray-di-o-lo-gist"
- prevalence: "pre-va-lence"
- regularizer: "re-gu-la-ri-zer"

# Before you start

1. Join ten minutes early. Test microphone and screen share.
2. Share the PDF window only, not the whole screen.
3. Open `main.pdf` in another window, for number questions.
4. References are pages 16 to 19, thank-you is 20, backup slides are pages 21 to 25:
   21 full RSNA metrics, 22 per-label zero-shot on SPIDER, 23 core equations,
   24 full training objective, 25 parameters and statistics. Type the page number and press Enter.
5. There is no laser pointer. Say the position out loud: "the Severe Recall row, last column, zero point four nine".
