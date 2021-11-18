# se.uu.statml

A comparison between the outputs of following regressions

**Logistic Regression**

LR = LogisticRegression(max_iter=10000)

LR.fit(X_train, y_train.values.ravel())

scores = cross_val_score(LR, X_train, y_train.values.ravel(), cv=2)

print(f"cross val scores: {scores}, test score: {LR.score(X_test, y_test)}")

**_cross val scores: [0.8543956  0.83471074], test score: 0.8846153846153846_**
