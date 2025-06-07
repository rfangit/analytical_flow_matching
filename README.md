# Analytical Flow Matching

<img src="\2D Examples\Linear Schedule - Other Distributions\imgs\checkerboard_example.png" style="width:100%; max-width:800px;"/>

This repository is a companion to the blog post **[“Optimal Flow-Matching”](https://rfangit.github.io/blog/2025/optimal_flow_matching/)**.

It showcases the optimal analytic solution to the flow-matching training objective, implemented for a variety of datasets including MNIST, CIFAR-10, and a subset of ImageNet. Although this solution is optimal in producing minimal MSEloss and does not require expensive training steps, there are two issues with it compared to a flow learned via deep learning:

- Evaluating the flow involves computing distances with all data points, which is computationally expensive for large datasets.
- It memorizes the data distribution, and only generates points at $t = 1$ that exist in the training data.

## 📂 File Structure

* **2D Examples/**: Demonstrations of flow-matching in two-dimensional settings. Contains work for arbitrary schedules, final variances, and a variety of toy two-dimensional problems.
* **CIFAR-10/**: Implementation with CIFAR-10 dataset.
* **Custom Dataset Example (Sine)/**: Example implementation for a custom dataset (a family of sine waves).
* **ImageNet/**: Implementation with ImageNet dataset.
* **MNIST/**: Implementation with MNIST dataset. This also contains a small flow-matching model trained on MNIST for comparison with the analytic results.
* **analytic\_flow\_funcs.py**: Functions implementing analytical flow-matching for arbitrary schedules and datasets.
* **image\_display\_funcs.py**: Utility functions for visualizing images and results.

The experiments demonstrate the efficacy of analytical solutions in flow-matching across different datasets. Visualizations are provided within each dataset's directory.

## 🧪 Reproducing Experiments

To explore the experiments:

1. **Install dependencies**:

   Install the required Python packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run notebooks**:

   Open any notebook under experiment directories (e.g., `MNIST/`, `CIFAR-10/`) to run the experiments interactively.

## Functionality

analytic_flow_funcs has support for:
- linear time flow with scalar time input (Appendix 4.2)
- linear time flow with tensor time input (eg, when flowing to M data points, you can specify M time values)
- arbitrary time flow with finite but uniform variance for data points (Appendix 4, partial implemenation - no support for different variances)
- linear time flow with distinct variances (Appendix 4.4)

It does not support:
- arbitrary time flow with distinct variances (Appendix 4.3)
