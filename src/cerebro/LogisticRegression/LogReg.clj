(ns cerebro.LogisticRegression.LogReg
  (:use [cerebro.Utils.utils]))

; Logistic regression is a probabilistic, linear classifier. It is parametrized
; by a weight matrix :math:`weights` and a bias vector :math:`bias`. 
; Classification is done by projecting data points onto a set of hyperplanes, 
; the distance to which is used to determine a class membership probability.

(defrecord LogReg [num-inputs num-outputs weights bias])

(defn make-log-reg [n-in n-out]
  (->LogReg n-in 
            n-out 
            (mapv vec (partition n-out (take (* n-out n-in) (repeat 0.0)))) 
            (take n-out (repeat 0.0))))	           

(defn softmax [x-coll]
  (let [_max (apply max x-coll)
        sum  (reduce + (map #(Math/exp (- % _max)) x-coll))]
    (map #(/ % sum) x-coll)))

(defn train [logreg x y lr]
  (let [px|y nil]
    px|y))

