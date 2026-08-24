# Speaking script, MIWAI 2026 paper 122

Online, live, 15 minutes. Target: **12 to 13 minutes of talking**, then questions.

Rules for delivery: short sentences, slow, pause at every `/`. Say numbers slowly. If you lose your place, look at the slide title and continue. Nobody minds a pause.

---

## Slide 1. Title (20 seconds)

> Hello everyone. / My name is Trung Kien Ha, / from Ho Chi Minh City University of Technology, Vietnam. / This is joint work with my advisor, Doctor Trong Nhan Phan.
>
> Today I will talk about / a two-branch model / for lumbar disc grading / and zero-shot label extension.

---

## Slide 2. Outline (15 seconds)

> I will cover four parts. / First the problem. / Then our method. / Then the experiments. / And finally the conclusion.

*(Do not read the list. Just say this and move on.)*

---

## Slide 3. Two open problems (1 minute)

> We work on lumbar spine MRI. / Doctors grade each disc by hand. / This is slow, / and different doctors often disagree.
>
> There are two open problems.
>
> The first one is class imbalance. / The most important class is "Severe". / But it is less than five percent of the data. / So a normal model almost never predicts it. / Its recall is close to zero. / That is the class doctors care about most.
>
> The second problem is a fixed label space. / A model is trained on one label set. / If we want a new label set, / we must build a new head / and train again.

---

## Slide 4. Contributions (1 minute)

> We have four contributions.
>
> First, a finding about transfer. / Attention helps the Severe class / on the same dataset. / But it hurts on a different dataset. / A frozen prior fixes this.
>
> Second, a two-branch design. / The two branches are complementary, / not redundant.
>
> Third, zero-shot label extension. / The label set becomes an input. / We do not retrain anything.
>
> Fourth, a training recipe for imbalance. / It lifts Severe recall / from twelve point three percent / to forty-eight point six percent. / About four times better.

---

## Slide 5. Architecture (2 minutes), the core slide, go slow

> This is our model. / One disc volume goes in. / Two branches read it.
>
> The top branch is the attention branch. / It is a three-D ResNet-34 / with CBAM attention. / CBAM means channel and spatial attention. / This branch looks at the shape of the disc. / It helps the model focus / on the small area where the lesion is.
>
> The bottom branch is the multimodal branch. / We use BiomedCLIP. / It is frozen. / We never train it. / It was pretrained / on millions of medical image and text pairs. / So it already knows a lot / about medical images.
>
> We take the two vectors. / We put them together. / A small MLP mixes them / into one vector.
>
> Now the important part. / On the right side, / we write the labels as text prompts. / The text encoder turns them into vectors. / We compare our image vector / with each label vector, / using cosine similarity. / The closest one is the prediction.
>
> Both backbones stay frozen. / We only train the small head. / That is about one point two million parameters, / out of two hundred sixty million.

---

## Slide 6. Class imbalance (1 minute)

> For the imbalance problem / we use three things together.
>
> First, focal loss. / It puts more weight on hard examples.
>
> Second, square-root class weights. / Without this, / the gradient ratio is about fifteen times. / With it, about four times. / Training is more stable.
>
> Third, oversampling. / We show Severe cases more often.
>
> Why does this work? / The frozen branch gives a stable prior / for the rare class. / The attention branch finds the small lesion.

---

## Slide 7. Zero-shot label extension (1 minute 30)

> This slide is the main idea of the paper.
>
> The encoder stays the same. / Only the head changes.
>
> We write a prompt for each label. / For example, / "a magnetic resonance image of lumbar disc herniation".
>
> The text encoder turns the prompt into a vector. / We compute cosine similarity / between the image vector / and each label vector. / We take the highest one.
>
> So a new label / is just a new prompt. / No new head. / No retraining.
>
> This is why we say / the label set is an input to the model, / not a fixed part of the architecture.

---

## Slide 8. Setup (45 seconds)

> We use two datasets.
>
> RSNA 2024 is for training / and for in-domain testing. / About nineteen hundred discs for validation.
>
> SPIDER is a different dataset, / from Dutch hospitals. / We never train on it. / We only use it to test transfer.
>
> We compare four models. / The SpineNetV2 backbone. / CBAM only. / Multimodal only. / And our full Hybrid.
>
> All results use three seeds, / on one RTX 4090.

---

## Slide 9. In-domain results (2 minutes), do not read every cell

> This is the main result table.
>
> Please look at the Mean F1 row. / SpineNetV2 gets zero point four two. / Our Hybrid gets zero point five three. / Both single branches are in between. / So the two branches together / are better than each one alone.
>
> Now the bottom part, / the Severe class. / This is what matters clinically.
>
> Severe recall goes from zero point one two / to zero point four nine. / About four times better.
>
> Severe F1 goes from zero point one five / to zero point three six.
>
> One number goes down. / Mean accuracy drops about nine points. / I want to be honest about this. / The baseline gets high accuracy / because it predicts "Normal" almost every time. / We trade some accuracy / to catch more Severe cases. / In the clinic, / missing a Severe case / is worse than a false alarm.

---

## Slide 10. Not guessing (1 minute)

> One reviewer asked a fair question. / The F1 numbers look low. / Is the model just guessing?
>
> We do not think so.
>
> Severe is only five percent of the data. / A random model would get / AUPRC around zero point zero five. / We get zero point three three. / That is about six point seven times better than chance. / And the AUC is zero point nine.
>
> The numbers look low / because we keep the hard setting. / Three classes. / The real five percent distribution. / And macro averaging, not accuracy.
>
> On the right / you can see Grad-CAM. / Without CBAM the model looks everywhere. / With CBAM it focuses on the canal.

---

## Slide 11. Zero-shot transfer to SPIDER (1 minute 15)

> Now the zero-shot test.
>
> We take the model trained on RSNA. / We change nothing. / We only write new prompts / for eight SPIDER labels. / The model has never seen these labels, / and never seen this dataset.
>
> For disc-shape labels, / our model gets zero point five nine, / against zero point four three / for BiomedCLIP alone. / So the RSNA training does transfer.
>
> But I should be honest here. / For labels far from the RSNA schema, / the off-the-shelf model is better. / Over all eight labels, / we get zero point three six, / and it gets zero point three nine.
>
> So the transfer works / when the new label is close / to what we trained on. / Not for everything.

---

## Slide 12. Take-aways (1 minute)

> Three take-aways.
>
> First, / the imbalance recipe / recovers the Severe class. / Recall about four times higher.
>
> Second, / labels can be an input. / A new label set is just new prompts. / No retraining.
>
> Third, / the frozen branch acts like a regularizer. / Attention alone drops / on a new dataset. / The frozen branch brings it back.
>
> One honest note. / We only ran three seeds. / So the p-values are exploratory. / It is not a powered test.
>
> Thank you. / I am happy to take questions.

---

# Questions: prepared answers

**Q. Three seeds is not enough for significance.**

> You are right. / We report it as exploratory. / With three seeds / we cannot claim a powered test. / More seeds is future work.

**Q. Why should cosine similarity work for a new prompt?**

> During training / we already score by cosine / against the RSNA text prompts. / So the image vector is pulled / toward the text space. / For a new label / it works if the label is close / to something we trained on. / Table four shows both sides. / We win on disc-shape labels / and we lose on distant labels. / We do not claim it works for everything.

**Q. What is the exact training loss?**

*(Go to backup slide 15.)*

> Focal loss on the cosine logits. / Plus a supervised contrastive term, / weight zero point one. / And the tasks are combined / by learned uncertainty weighting.

**Q. Is it fast enough for clinical use?**

> Training is cheap. / We only train one point two million parameters. / But inference is not light. / Both frozen backbones still run. / About sixty-six milliseconds per disc.

**Q. Why BiomedCLIP and not another model?**

> It is public, / and it gives a shared image and text space. / That is what we need for prompts. / Comparing other medical vision-language models / is future work.

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
4. Backup slides are pages 15 to 18. Type the page number and press Enter.
5. There is no laser pointer. Say the position out loud: "the Severe Recall row, last column, zero point four nine".
