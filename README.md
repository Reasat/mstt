**MSTT-199: MRI Dataset for Musculoskeletal Soft Tissue
Tumor Segmentation** 

Accurate soft tissue tumor segmentation is vital for assessing tumor size,
location, diagnosis, and response to treatment, thereby influencing patient outcomes.
However, segmentation of these tumors requires clinical expertise, and an automated
segmentation model would save valuable time for both clinician and
patient. Training an automatic model requires a large dataset of annotated
images. In this work, we describe the collection of an MR imaging dataset of 199 musculoskeletal soft tissue tumor from 199 patients. We trained segmentation models on this dataset and then benchmarked them on a publicly available dataset. Our
model achieved the state-of-the-art dice score of 0.79 out of the box
without any fine tuning, which shows the diversity and utility of our
curated dataset. We analyzed the model predictions and found that its
performance suffered on fibrous and vascular tumors due to their
diverse anatomical location, size, and intensity heterogeneity.

Datasets and Model weights will be published after the acceptance of the paper.
