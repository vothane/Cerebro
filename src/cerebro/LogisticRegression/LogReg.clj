(ns cerebro.LogisticRegression.LogReg
  (:use [cerebro.Utils.utils])
  (:refer-clojure :exclude [min max])
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]))

; Logistic regression is a probabilistic, linear classifier. It is parametrized
; by a weight matrix :math:`weights` and a bias vector :math:`bias`.
; Classification is done by projecting data points onto a set of hyperplanes,
; the distance to which is used to determine a class membership probability.

(defrecord LogReg [N num-inputs num-outputs weights bias])

(defn make-log-reg [N n-in n-out]
  (->LogReg N n-in n-out (zero-matrix n-out n-in) (zero-vector n-out)))

(defn softmax [x]
  (let [max (emax x)
        x   (emap #(Math/exp (- % max)) x)
        sum (esum x)]
    (emap #(/ % sum) x)))

(defn train [logreg x y lr]
  (let [f    (fn [v] (esum (add v x)))
        px|y (add (map #(esum %) (:weights logreg)) (:bias logreg))
        dy   (sub y (softmax px|y))
        f    (fn [[i j] v] (+ v (/ (* lr (mget dy i) (mget x j)) (:N logreg))))]
    (-> logreg
        (assoc :weights (emap-indexed f (:weights logreg)))
        (assoc :bias (add (:bias logreg) (emap #(/ (* lr %) (:N logreg)) dy))))))

(defn predict [logreg x]
  (let [f (fn [v] (esum (add v x)))
        y (add (map #(esum %) (:weights logreg)) (:bias logreg))]
    (softmax y)))
