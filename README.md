# Skin lesion segmentation with a U-Net

A U-Net trained from scratch to outline skin lesions in dermoscopy photos from HAM10000 (10,015 photos of 7,470 lesions, each with a lesion mask). The model takes a photo and marks every pixel as lesion or skin.

Version 2 asks one question about version 1: was its test score inflated because photos of the same lesion ended up in both training and test?

## What version 1 did and why it needed checking

Version 1 trained on a T4 GPU at 256 by 256 pixels and reported a test Dice of 0.9115. It split the 10,015 photos at random, 80/10/10. Many lesions in HAM10000 were photographed more than once, so a random split by photo puts some lesions on both sides. In the version 1 split, 402 of the 1,002 test photos show a lesion that also has a photo in training. If those photos are near copies, the test is partly a memory test.

Dice measures overlap between the predicted outline and the true one: 2 times the shared pixels, divided by the predicted pixels plus the true pixels. 1 means identical, 0 means no overlap. Below, a "failure" is a photo with Dice under 0.5.

## What version 2 found

Splitting by lesion did not lower the score. I trained the same model six times: three random seeds on the version 1 style split (by photo), and three on a split by lesion, where every photo of a lesion stays on one side.

| Split | Seed 1 | Seed 2 | Seed 42 | Mean of seeds |
|---|---|---|---|---|
| By photo (version 1 style) | 0.934 | 0.933 | 0.930 | 0.932 |
| By lesion | 0.941 | 0.931 | 0.937 | 0.936 |

The lesion split came out slightly higher on two seeds and 0.002 lower on the third. Those gaps are smaller than the spread between seeds of the same split (0.931 to 0.941 for the lesion split), so the honest split and the leaky one score the same within noise. Failures stayed under 1% in every run (0.4% to 0.8%).

Two checks explain why the overlap did not help:

* Two photos of one lesion are not close copies. For the 402 test photos with a sibling in training, copying the sibling's training mask straight onto the test photo, with no model at all, scores a mean Dice of 0.644, far below the model.
* The model did no better on lesions it had seen. Among nevi (moles), the largest group, test photos of lesions seen in training and of unseen lesions both scored 0.949 (194 and 476 photos). I compared within nevi because the seen and unseen groups have different diagnosis mixes.

So version 1's split was the wrong design, but it did not inflate the result in a way these runs can detect.

## How it compares with methods that learn nothing

On the seed 42 lesion split test set (1,002 photos):

| Method | Mean Dice (95% interval) | Failures |
|---|---|---|
| Same fixed mask for every photo, from how often each pixel is lesion in training | 0.640 (0.627 to 0.653) | 21.7% |
| Grey level threshold (Otsu), keep the region nearest the centre, fill holes | 0.750 (0.734 to 0.767) | 9.7% |
| U-Net | 0.937 (0.931 to 0.943) | 0.8% |

The threshold method already gets many photos roughly right. The U-Net's gain shows most in the failures: 9.7% of photos for the threshold, 0.8% for the U-Net.

## Where it is weaker

Mean Dice by diagnosis, lesion split, all three seeds pooled:

| Diagnosis | Test photos per seed | Mean Dice | Failures |
|---|---|---|---|
| Nevus (nv) | 671 | 0.953 | 0.3% |
| Melanoma (mel) | 111 | 0.937 | 0.3% |
| Benign keratosis (bkl) | 110 | 0.919 | 0.0% |
| Dermatofibroma (df) | 11 | 0.879 | 0.0% |
| Actinic keratosis (akiec) | 32 | 0.864 | 3.1% |
| Vascular (vasc) | 14 | 0.858 | 4.7% |
| Basal cell carcinoma (bcc) | 51 | 0.836 | 3.9% |

Basal cell carcinoma, actinic keratosis and vascular lesions are the weak spots. The worst test photo (Dice 0.20, in `results/examples.png`) is an actinic keratosis that runs off the edge of the image; the model outlined only the small dark centre. Dermatofibroma and vascular lesions have 11 to 14 test photos per seed, so their numbers are rough.

## Limits

* Inputs are 128 by 128 pixels and the model has 32 base channels (7,763,041 parameters), so it could train on GitHub's CPU runners. Version 1 used 256 by 256 and 64 base channels (31,037,633 parameters). Masks at lower resolution have smoother edges, so 0.932 here and 0.9115 in version 1 are not a fair comparison and should not be read as an improvement.
* Training ran a fixed 20 epochs, as in version 1. In two of the six runs the best validation loss came at epoch 20, so longer training might still help.
* Everything is tested on HAM10000 only. There is no test on photos from another clinic or camera.
* Dice says nothing about diagnosis. This model outlines lesions; it does not say whether one is cancer.

## Files

* `src/skinseg/`: data download and caching (`data.py`), the two splits (`splits.py`), the U-Net (`model.py`), training (`train.py`), the two no learning methods (`baselines.py`), metrics with confidence intervals that resample whole lesions (`metrics.py`) and the report (`report.py`).
* `results/RESULTS.md`: every table above plus IoU, medians and intervals for each run. `results/per_image.csv` has the score for every test photo in every run.
* `models/unet32_lesion_split.pth`: weights from the seed 42 lesion split run.
* `.github/workflows/train.yml` and `finish.yml`: the runs that produced these results. Training saves its state after each epoch and continues in a new job if a runner is too slow to finish within GitHub's time limit.
* `skin_lesion_segmentation.ipynb`: the original version 1 notebook.
* `space/`: the Hugging Face Space demo. `app.py` loads the seed 42 lesion split weights from this repository; `unet.py` is a copy of `src/skinseg/model.py` so the Space runs without installing the package.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m skinseg.data data                       # downloads HAM10000 from Harvard Dataverse
python -m skinseg.train --split lesion --seed 42 --out runs/lesion-s42
python -m skinseg.report --runs runs --data data --out results
pytest
```

One training run took between about 3.5 and 6.5 hours on a 4 core GitHub runner, depending on the machine.

## Demo

Hugging Face Space: https://huggingface.co/spaces/samurvivor-07/skin-lesion-segmentation
