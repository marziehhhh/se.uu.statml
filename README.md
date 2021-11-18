# se.uu.statml

A comparison between the outputs of following regressions

**LOGISTIC REGRESSION**

LR = LogisticRegression(max_iter=10000)

LR.fit(X_train, y_train.values.ravel())

scores = cross_val_score(LR, X_train, y_train.values.ravel(), cv=2)

print(f"cross val scores: {scores}, test score: {LR.score(X_test, y_test)}")

**_cross val scores: [0.8543956  0.83471074], test score: 0.8846153846153846_**

**KNEIGHBORS**

KNN = KNeighborsClassifier()

KNN.fit(X_train, y_train.values.ravel())

scores = cross_val_score(KNN, X_train, y_train.values.ravel(), cv=2)

print(f"cross val scores: {scores}, test score: {KNN.score(X_test, y_test)}")

**_cross val scores: [0.75       0.75482094], test score: 0.8076923076923077_**

