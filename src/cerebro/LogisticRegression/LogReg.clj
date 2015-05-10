(ns cerebro.LogisticRegression.LogReg)

; Logistic regression is a probabilistic, linear classifier. It is parametrized
; by a weight matrix :math:`weights` and a bias vector :math:`bias`. 
; Classification is done by projecting data points onto a set of hyperplanes, 
; the distance to which is used to determine a class membership probability.

(defrecord LogReg [N
                   num-inputs
                   num-outputs
                   weights
                   bias])

(defn make-log-reg [n n-in n-out]
  (->LogReg n 
            n-in 
            n-out 
            (partition n-out (take (* n-out n-in) (repeat 0.0))) 
            (take n-out (repeat 0.0))))	           

(defn softmax [x-coll]
  (let [_max (apply max x-coll)
        sum  (reduce + (map #(Math/exp (- % _max)) x-coll))]
    (map #(/ % sum) x-coll)))

(defn train [logreg x y lr]
  (let [mults (map #(map (fn [a b] (* a b)) x %) (:weights logreg))
        px|y  (map #(reduce + %) mults)
        px|y  (map #(+ %1 %2) px|y (:bias logreg))
        px|y  (softmax px|y)]
    px|y))

