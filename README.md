5. Scaling Experiment (Mandatory Experiment 1)
Goal: Understand how feature scaling affects distance-based model performance versus tree-based models.

Why is KNN sensitive to scaling? KNN relies on Euclidean distance to find the closest data points. Without scaling, features with massive variances (like Time and Amount) completely overshadow the PCA-transformed features (V1 to V28). This breaks the distance calculation, rendering the model almost entirely unable to detect fraud.

Why is Decision Tree less sensitive? Decision Trees split nodes based on single-feature thresholds independently. Because the model looks at one feature at a time to make a split, monotonic scaling (like Standard Scaler) does not change the mathematical logic or order of the splits, making trees naturally scale-invariant.

6. Hyperparameter Experiment (Mandatory Experiment 2)
Goal: Analyze the effect of the max_depth hyperparameter on a Decision Tree to observe the balance between underfitting and overfitting.

Did overfitting occur? Yes. As the tree depth increases to None (fully expanded), we see a significant gap widen between the Validation PR-AUC (0.5780) and Test PR-AUC (0.4766). This indicates the model begins memorizing the training data rather than generalizing.

Which value provides the best balance? A moderate depth like max_depth=10 provides a safer balance for fraud detection. It prevents extreme overfitting while maintaining a much higher Recall (~75% vs ~65% for a fully expanded tree), which is critical for catching fraudulent transactions.

7. Impact of Classification Threshold (Mandatory Experiment 3)
Goal: Investigate how changing the default classification threshold (0.5) affects fraud detection metrics using Logistic Regression at thresholds of 0.3, 0.5, and 0.7.

What happens to Recall when the threshold decreases? Lowering the threshold generally increases Recall, because the model becomes more permissive and flags more transactions as fraud, catching cases it might have otherwise missed.

What happens to Precision? Decreasing the threshold drastically reduces Precision. Because the model becomes highly sensitive, it generates a massive amount of false positives (legitimate transactions incorrectly flagged as fraud).

Which threshold would you recommend for a fraud detection system? A lower threshold (e.g., 0.3 or 0.5) is recommended.

What trade-off does your chosen threshold create? By choosing a lower threshold, we accept the trade-off of lowering our Precision (more false alarms) to maximize our Recall. In banking, the financial and reputational cost of missing a fraudulent transaction (False Negative) is exponentially worse than the operational cost of manually reviewing a false positive.

8. Additional Bonus Experiments
Handling Class Imbalance: We tested a baseline Logistic Regression model against models using class_weight='balanced' and Random Oversampling. The baseline model missed a massive amount of fraud (Recall of ~58%). Both imbalance handling strategies massively improved Recall (jumping to ~87%), proving that standard models are heavily biased toward the majority class without intervention.

Model Leaderboard: We evaluated multiple model families. Random Forest emerged as the most robust model for this dataset, achieving an excellent balance of Validation F1 (0.8591) and Test PR-AUC (0.8022), handling the non-linear complexities of the data much better than linear models.

Learning Curve Analysis: A 5-Fold Stratified Cross-Validation learning curve was plotted. The PR-AUC scores stabilized tightly around 0.75 as the training size reached its maximum. Because the validation score plateaued and closely tracked the training score, the model generalizes well and does not appear to be suffering from high variance.