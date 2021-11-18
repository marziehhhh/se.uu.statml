# se.uu.statml

A comparison between the outputs of following regressions

**LOGISTIC REGRESSION**

LR = LogisticRegression(max_iter=10000)

LR.fit(X_train, y_train.values.ravel())

scores = cross_val_score(LR, X_train, y_train.values.ravel(), cv=2)

print(f"cross val scores: {scores}, test score: {LR.score(X_test, y_test)}")

**_cross val scores: [0.8543956  0.83471074], test score: 0.8846153846153846_**

<===============================================================================>

**KNEIGHBORS**

KNN = KNeighborsClassifier()

KNN.fit(X_train, y_train.values.ravel())

scores = cross_val_score(KNN, X_train, y_train.values.ravel(), cv=2)

print(f"cross val scores: {scores}, test score: {KNN.score(X_test, y_test)}")

**_cross val scores: [0.75       0.75482094], test score: 0.8076923076923077_**

<===============================================================================>

**LDA**

lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train.values.ravel())

scores = cross_val_score(lda, X_train, y_train.values.ravel(), cv=5)

print(f"cross val scores: {scores}, test score: {lda.score(X_test, y_test)}")

**_cross val scores: [0.84246575 0.84931507 0.85517241 0.8        0.86206897], test score: 0.8782051282051282_**

<===============================================================================>

**QDA**

qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train, y_train)

scores = cross_val_score(qda, X_train, np.array(y_train), cv=5)

print(f"cross val scores: {scores}, test score: {qda.score(X_test, y_test)}")

**_cross val scores: [0.89726027 0.82191781 0.85517241 0.84827586 0.89655172], test score: 0.7596153846153846_**

<===============================================================================>


