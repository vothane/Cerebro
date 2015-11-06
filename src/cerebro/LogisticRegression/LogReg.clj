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
  (->LogReg N
            n-in
            n-out
            (mapv vec (partition n-out (take (* n-out n-in) (repeat 0.0))))
            (vec (take n-out (repeat 0.0)))))

(defn softmax [x]
  (let [max (emax x)
        x   (emap #(Math/exp (- % max)) x)
        sum (esum x)]
    (emap #(/ % sum) x)))

(defn train [logreg x y lr]
  (let [f    (fn [v] (esum (add v x)))
        px|y (emap #(+ %1 %2) (emap f (:weights logreg)) (:bias logreg))
        dy   (sub y (softmax px|y))
        f    (fn [[i j] v] (+ v (/ (* lr (mget dy i) (mget x j)) (:N logreg))))]
    (-> logreg
        (assoc :weights (emap-indexed f (:weights logreg)))
        (assoc :bias (add (:bias logreg) (emap #(/ (* lr %) (:N logreg)) dy))))))


