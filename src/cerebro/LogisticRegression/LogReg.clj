(ns cerebro.LogisticRegression.LogReg
  (:use [cerebro.Utils.utils]))

; Logistic regression is a probabilistic, linear classifier. It is parametrized
; by a weight matrix :math:`weights` and a bias vector :math:`bias`.
; Classification is done by projecting data points onto a set of hyperplanes,
; the distance to which is used to determine a class membership probability.

(defn softmax [x]
  (let [m (apply max x)
        x (map #(Math/exp (- % max)) x)
        s (reduce + x)]
    (map #(/ % s) x)))

