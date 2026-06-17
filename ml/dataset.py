"""Labeled training data for the sentence rhetorical-role classifier.

Four roles common in scientific writing:
  BACKGROUND  - motivation, prior work, the problem
  METHOD      - what was done / how the approach works
  RESULT      - measured outcomes, numbers, findings
  CONCLUSION  - takeaways, implications, future work

This is a compact, hand-labeled seed set. It's enough for a TF-IDF + Logistic
Regression model to learn the keyword/phrasing patterns of each role, and it's
easy to extend - just add (sentence, label) pairs.
"""

DATA = [
    # ---------------- BACKGROUND ----------------
    ("Recurrent neural networks have long been the dominant approach to sequence modeling.", "BACKGROUND"),
    ("Prior work has shown that attention mechanisms improve translation quality.", "BACKGROUND"),
    ("Despite recent progress, scaling these models remains a major challenge.", "BACKGROUND"),
    ("Understanding protein folding is a fundamental problem in biology.", "BACKGROUND"),
    ("Existing methods struggle to capture long-range dependencies in text.", "BACKGROUND"),
    ("Many real-world systems require low-latency inference under tight constraints.", "BACKGROUND"),
    ("The problem of catastrophic forgetting has been studied for decades.", "BACKGROUND"),
    ("Traditional convolutional architectures process inputs sequentially.", "BACKGROUND"),
    ("Several authors have previously proposed graph-based representations.", "BACKGROUND"),
    ("It is well known that deep networks are prone to overfitting on small datasets.", "BACKGROUND"),
    ("Transfer learning has become increasingly important in low-resource settings.", "BACKGROUND"),
    ("The motivation for this study stems from limitations in current retrieval systems.", "BACKGROUND"),
    ("Climate models have historically relied on coarse spatial resolution.", "BACKGROUND"),
    ("Recent advances in hardware have enabled training of much larger models.", "BACKGROUND"),
    ("However, little attention has been paid to the interpretability of these systems.", "BACKGROUND"),
    ("This area has attracted growing interest over the past few years.", "BACKGROUND"),
    ("The need for reliable uncertainty estimates motivates our investigation.", "BACKGROUND"),
    ("Earlier approaches typically depend on hand-engineered features.", "BACKGROUND"),

    # ---------------- METHOD ----------------
    ("We propose a novel architecture based entirely on self-attention.", "METHOD"),
    ("Our model is trained using stochastic gradient descent with momentum.", "METHOD"),
    ("We introduce a regularization term to penalize large activations.", "METHOD"),
    ("The encoder consists of six identical layers with residual connections.", "METHOD"),
    ("We fine-tune the pretrained model on the downstream task.", "METHOD"),
    ("To evaluate the approach, we design a controlled experiment.", "METHOD"),
    ("The algorithm computes attention weights using scaled dot products.", "METHOD"),
    ("We implement the system in PyTorch and train on eight GPUs.", "METHOD"),
    ("Hyperparameters were selected via grid search on the validation set.", "METHOD"),
    ("Our method partitions the input into fixed-size patches.", "METHOD"),
    ("We apply dropout with probability 0.1 after each sublayer.", "METHOD"),
    ("The loss function combines cross-entropy with a contrastive term.", "METHOD"),
    ("We preprocess the corpus by tokenizing and lowercasing the text.", "METHOD"),
    ("Each model is trained for one hundred thousand steps.", "METHOD"),
    ("We use a cosine learning-rate schedule with linear warmup.", "METHOD"),
    ("The proposed framework integrates retrieval with generation.", "METHOD"),
    ("We augment the dataset using random cropping and flipping.", "METHOD"),
    ("Our approach encodes each document into a dense vector representation.", "METHOD"),

    # ---------------- RESULT ----------------
    ("Our model achieves a BLEU score of 41.8 on the translation benchmark.", "RESULT"),
    ("Accuracy improved by 3.2 percentage points over the baseline.", "RESULT"),
    ("The proposed method reduces inference latency by 40 percent.", "RESULT"),
    ("We observe a significant decrease in validation loss after pretraining.", "RESULT"),
    ("The model outperforms all prior systems on five of six tasks.", "RESULT"),
    ("Table 2 reports the F1 scores across all evaluated datasets.", "RESULT"),
    ("Performance degrades sharply when the context length exceeds 512 tokens.", "RESULT"),
    ("Our experiments show a strong correlation between depth and accuracy.", "RESULT"),
    ("The error rate dropped from 8.4 percent to 5.1 percent.", "RESULT"),
    ("Figure 3 illustrates that convergence is faster with our optimizer.", "RESULT"),
    ("We find that larger batch sizes yield only marginal gains.", "RESULT"),
    ("The ablation study confirms that each component contributes to performance.", "RESULT"),
    ("Recall increased substantially while precision remained stable.", "RESULT"),
    ("Our system achieves state-of-the-art results on the GLUE benchmark.", "RESULT"),
    ("The model attained 92 percent accuracy on the held-out test set.", "RESULT"),
    ("Throughput scaled almost linearly with the number of workers.", "RESULT"),
    ("Quantitative results demonstrate consistent improvements across seeds.", "RESULT"),
    ("The measured speedup was 2.3 times relative to the dense baseline.", "RESULT"),

    # ---------------- CONCLUSION ----------------
    ("In conclusion, attention-based models offer a compelling alternative.", "CONCLUSION"),
    ("These findings suggest that self-attention generalizes well across tasks.", "CONCLUSION"),
    ("We believe this approach opens new directions for future research.", "CONCLUSION"),
    ("Overall, our results highlight the importance of large-scale pretraining.", "CONCLUSION"),
    ("Future work will explore extending the method to multimodal inputs.", "CONCLUSION"),
    ("This work demonstrates that simpler architectures can be highly effective.", "CONCLUSION"),
    ("Taken together, the evidence supports our central hypothesis.", "CONCLUSION"),
    ("We hope these insights will guide the design of next-generation systems.", "CONCLUSION"),
    ("Our study has implications for both research and practical deployment.", "CONCLUSION"),
    ("In summary, the proposed framework is both efficient and accurate.", "CONCLUSION"),
    ("Further investigation is needed to assess robustness in the wild.", "CONCLUSION"),
    ("These results pave the way for more interpretable models.", "CONCLUSION"),
    ("We conclude that the method is well suited for low-resource settings.", "CONCLUSION"),
    ("Ultimately, this contribution advances the state of the field.", "CONCLUSION"),
    ("The broader impact of this technology warrants careful consideration.", "CONCLUSION"),
    ("Our findings underscore the value of combining retrieval and generation.", "CONCLUSION"),
    ("Moving forward, we plan to release the code and trained models.", "CONCLUSION"),
    ("This suggests a promising avenue for reducing computational cost.", "CONCLUSION"),
]

LABELS = ["BACKGROUND", "METHOD", "RESULT", "CONCLUSION"]
