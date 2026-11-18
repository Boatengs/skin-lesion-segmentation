# Results

U-Net, 7,763,041 parameters, 128 by 128 inputs, 20 epochs on CPU. Version 1 used the same layout with 31,037,633 parameters at 256 by 256 on a GPU and reported test Dice 0.9115.

Dice is the overlap between the predicted and the true lesion outline: 2 x shared pixels / (predicted + true pixels), from 0 (no overlap) to 1 (identical). A failure is an image with Dice below 0.5. Intervals are 95% and resample whole lesions.

## Every run

| Split | Seed | Test images | Seen lesion in training | Mean Dice (95% interval) | Median Dice | Mean IoU | Failures (95% interval) | Best epoch |
|---|---|---|---|---|---|---|---|---|
| image | 1 | 1002 | 379 | 0.934 (0.928 to 0.939) | 0.964 | 0.885 | 0.4% (0.1% to 0.8%) | 20 |
| image | 2 | 1002 | 382 | 0.933 (0.927 to 0.939) | 0.966 | 0.885 | 0.7% (0.2% to 1.3%) | 19 |
| image | 42 | 1002 | 402 | 0.930 (0.923 to 0.936) | 0.966 | 0.881 | 0.7% (0.3% to 1.3%) | 16 |
| lesion | 1 | 1002 | 0 | 0.941 (0.934 to 0.946) | 0.969 | 0.896 | 0.5% (0.1% to 1.0%) | 20 |
| lesion | 2 | 1002 | 0 | 0.931 (0.925 to 0.937) | 0.963 | 0.881 | 0.6% (0.1% to 1.2%) | 18 |
| lesion | 42 | 1002 | 0 | 0.937 (0.931 to 0.943) | 0.963 | 0.890 | 0.8% (0.3% to 1.5%) | 15 |

Image split minus lesion split, mean Dice, by seed: seed 1: -0.007, seed 2: +0.002, seed 42: -0.008.

## How alike are two photos of one lesion

In the version 1 split (image split, seed 42), 402 of 1002 test images show a lesion that also has a photo in training. Copying that training photo's mask, with no model at all, scores mean Dice 0.644 (median 0.669).

| Test images (image split, seed 42) | Images | Model mean Dice |
|---|---|---|
| Lesion seen in training | 402 | 0.921 |
| Lesion not seen | 600 | 0.936 |
| Nevi only, seen | 194 | 0.949 |
| Nevi only, not seen | 476 | 0.949 |

The seen and unseen groups differ in diagnosis mix, so the nevus only rows are the fairer comparison.

## Methods with no learning (seed 42 test sets)

| Split | Method | Mean Dice (95% interval) | Failures |
|---|---|---|---|
| image | prior | 0.641 (0.629 to 0.652) | 19.8% |
| image | otsu | 0.735 (0.719 to 0.751) | 13.4% |
| lesion | prior | 0.640 (0.627 to 0.653) | 21.7% |
| lesion | otsu | 0.750 (0.734 to 0.767) | 9.7% |

prior: one fixed mask from how often each pixel is lesion in training. otsu: grey level threshold, region nearest the centre, holes filled. See `baselines.py`.

## By diagnosis (all seeds pooled)

| Diagnosis | Split | Test images per seed | Mean Dice | Failures |
|---|---|---|---|---|
| akiec | image | 30 | 0.872 | 1.1% |
| akiec | lesion | 32 | 0.864 | 3.1% |
| bcc | image | 53 | 0.814 | 4.3% |
| bcc | lesion | 51 | 0.836 | 3.9% |
| bkl | image | 118 | 0.904 | 0.3% |
| bkl | lesion | 110 | 0.919 | 0.0% |
| df | image | 14 | 0.894 | 0.0% |
| df | lesion | 11 | 0.879 | 0.0% |
| mel | image | 115 | 0.933 | 0.6% |
| mel | lesion | 111 | 0.937 | 0.3% |
| nv | image | 654 | 0.952 | 0.3% |
| nv | lesion | 671 | 0.953 | 0.3% |
| vasc | image | 14 | 0.861 | 2.3% |
| vasc | lesion | 14 | 0.858 | 4.7% |
