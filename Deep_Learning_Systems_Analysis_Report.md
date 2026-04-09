# Deep Learning Systems Analysis Report

## Report Overview
This project addresses a supervised **image classification** problem using the Fashion-MNIST dataset and PyTorch. I implemented a baseline convolutional neural network (CNN) and compared it against a deeper CNN variant to test whether increasing architectural depth improved performance on grayscale clothing images. The focus was on correct implementation, controlled experimentation, and honest interpretation of the observed results.

## Dataset and Task Description
Fashion-MNIST contains 70,000 grayscale images of clothing items across 10 categories, with 60,000 training images and 10,000 test images at a resolution of 28×28 pixels (Xiao et al., 2017). This makes it a suitable benchmark for image-based deep learning because the task depends on learning visual patterns such as edges, silhouettes, and local textures.

In the notebook, the dataset is downloaded automatically through `torchvision.datasets.FashionMNIST(..., download=True)`, so access instructions are documented directly in the code. For a faster and reproducible local experiment, I used a fixed random subset of 15,000 examples from the original training split, divided into **12,000 training** and **3,000 validation** images, while keeping the full **10,000-image test set** for final evaluation.

## Model Architecture and Design Decisions
The baseline model is a compact CNN with two convolutional blocks followed by a fully connected classifier:

- `Conv2d(1→16) + ReLU + MaxPool`
- `Conv2d(16→32) + ReLU + MaxPool`
- `Flatten → Linear(32×7×7 → 128) + ReLU → Linear(128 → 10)`

A CNN is a good architectural match for this task because convolutional filters are designed to capture local spatial structure in images more efficiently than a fully connected network (LeCun et al., 1998; Goodfellow et al., 2016). I used **cross-entropy loss** for multi-class classification and the **Adam optimizer** with a learning rate of `0.001`, which is a practical default for stable training in many deep learning problems (Goodfellow et al., 2016; Paszke et al., 2019).

## Experimental Comparison
The comparison was designed as a **controlled architecture experiment**. I changed exactly one major factor: the number of convolutional layers.

- **Baseline model:** two convolutional layers
- **Experimental model:** three convolutional layers, with the added block increasing feature depth from 32 to 64 channels

Everything else stayed constant: dataset split, batch size (`128`), optimizer (`Adam`), loss function (`CrossEntropyLoss`), learning rate (`0.001`), and number of training epochs (`4`). This makes it reasonable to attribute any performance difference primarily to the added depth.

## Results and Interpretation
The deeper CNN performed slightly better than the baseline across all headline metrics:

| Model | Best Validation Accuracy | Test Accuracy | Macro F1 | Test Loss |
|---|---:|---:|---:|---:|
| Deeper CNN | 0.8647 | 0.8626 | 0.8606 | 0.3887 |
| Baseline CNN | 0.8643 | 0.8589 | 0.8571 | 0.4004 |

The performance gain was modest rather than dramatic, which suggests that the extra convolutional layer helped the model extract somewhat richer features, but the task was already reasonably well handled by the smaller network. The training and validation curves were stable for both models, and neither showed severe overfitting during the four training epochs.

A concrete example of model behavior appears in the class-level results: the best model performed extremely well on **Trouser** (`F1 = 0.972`) and **Bag** (`F1 = 0.954`), but much worse on **Shirt** (`F1 = 0.634`, `recall = 0.567`). This indicates that visually similar upper-body categories remained the hardest part of the problem, which is consistent with the limited resolution and grayscale-only format of the images.

## Limitations and Risks
Several limitations affect the strength of the conclusions:

1. The model was trained on a reduced training subset for faster execution, so the results may understate what the same architecture could achieve with the full dataset.
2. Only one main architectural change was tested, so the experiment does not explore other potentially important variables such as regularization, augmentation, or optimizer changes.
3. The images are low-resolution and grayscale, which removes color information that could help distinguish certain garment categories.
4. The evaluation reflects a single random split and a short training schedule, so the observed gap between the two models should be interpreted as suggestive rather than definitive.

## Ethical and Responsible Use
Although Fashion-MNIST is a benchmark dataset rather than a real deployed product pipeline, similar image-classification systems could be used in retail, inventory management, or automated recommendation settings. Misclassifications could lead to poor customer experiences, inventory errors, or biased downstream decisions if certain clothing styles are represented unevenly. More broadly, benchmark datasets may not capture the full diversity of real-world apparel, cultural clothing, or accessibility-related use cases, so responsible deployment would require better data coverage, monitoring, and human oversight (Goodfellow et al., 2016).

## Future Improvements
If more time and compute were available, I would improve the system in the following ways:

- train on the full 60,000-image training set rather than a smaller subset
- add image augmentation and regularization to improve robustness
- tune learning rate, epoch count, and batch size more systematically
- test a stronger architecture such as a residual CNN
- inspect more failure cases to understand why classes such as **Shirt** and **Pullover** remain difficult to separate

## References
Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org/

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324. https://doi.org/10.1109/5.726791

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*. https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv*. https://arxiv.org/abs/1708.07747
