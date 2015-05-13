(ns cerebro.LogisticRegression.LogReg
  (:use [cerebro.Utils.utils]
        [clojure.core.matrix])
  (:require [clojure.core.matrix.operators :as M]
            [clojure.math.numeric-tower :as math]))

; Logistic regression is a probabilistic, linear classifier. It is parametrized
; by a weight matrix :math:`weights` and a bias vector :math:`bias`. 
; Classification is done by projecting data points onto a set of hyperplanes, 
; the distance to which is used to determine a class membership probability.

(defrecord LogReg [num-inputs num-outputs weights bias])

(defn make-log-reg [n-in n-out]
  (->LogReg n-in 
            n-out 
            (mapv vec (partition n-out (take (* n-out n-in) (repeat 0.0))))
            (vec (take n-out (repeat 0.0)))))	           

(defn softmax [matrix]
  (let [_max   (apply max (flatten matrix))
        matrix (M/- matrix _max)]
    matrix))

(defn train [logreg x y lr]
  (let [px|y (M/+ (dp x (:weights logreg)) (:bias logreg))
        dy   (M/- y px|y)]
    (println px|y)))

