(ns cerebro.LogisticRegression.LogReg
  (:use [cerebro.Utils.utils]))

; Logistic regression is a probabilistic, linear classifier. It is parametrized
; by a weight matrix :math:`weights` and a bias vector :math:`bias`.
; Classification is done by projecting data points onto a set of hyperplanes,
; the distance to which is used to determine a class membership probability.

;; API functions
(defn train [logreg x y lr] ((:train logreg) x y lr))
(defn logreg->map [logreg] (:->map logreg))


(declare softmax)

(defn LogReg [weights bias n]
  {:train (fn [x y lr]
            (let [p-x|y (->> (map #(reduce + (map * x %)) weights)
                             (map + bias)
                             (softmax)) 
                  dy (map - y p-x|y)
                  calc-weight (fn [W i j] (-> (* lr dy_i x_j)
                                              (/ n)
                                              (+ w_ij)))
                  W (reduce 
                      (fn [weights [i j]] (put weights i j calc-weight)) 
                      weights
                      (for [i (range-rows weights) j (range-cols weights)] [i j]))
                  b (mapv (fn [bias_i dy_i] (+ bias_i (/ (* lr dy_i) n))) bias dy)]
          (LogReg W b n)))

   :->map {:weights weights :bias bias :n n}
  })

    ;; LogReg helper functions
    (defn softmax [x]
      (let [m (apply max x)
            x (map #(Math/exp (- % m)) x)
            s (reduce + x)]
        (mapv #(/ % s) x)))
